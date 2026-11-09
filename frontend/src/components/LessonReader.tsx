import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import type { Lesson, LessonMeta } from "../lib/api";
import { KIND_LABEL, sections as readSections, symbols as readSymbols, terms as readTerms } from "../lib/lessonOutline";
import { splitPart, titleOf } from "../lib/parts";
import Markdown from "./Markdown";

type Props = {
  lesson: Lesson;
  /** Every lesson, for the titles of neighbours and prerequisites. */
  lessons: LessonMeta[];
  /** Opens the exercise. Absent when the lesson has none. */
  onStartExercise?: () => void;
};

const FOCUS_KEY = "speech.reader.focus";
/**
 * A section is the one being read once its top passes this line: a third of
 * the way down the pane, where the eye rests, and never above the sticky bar.
 */
function readLine(pane: HTMLElement): number {
  return Math.max(120, pane.clientHeight * 0.3);
}

function rememberedFocus(): boolean {
  try {
    return window.localStorage.getItem(FOCUS_KEY) === "1";
  } catch {
    return false;
  }
}

export default function LessonReader({ lesson, lessons, onStartExercise }: Props) {
  const scroller = useRef<HTMLElement>(null);
  const [progress, setProgress] = useState(0);
  const [current, setCurrent] = useState(-1);
  const [read, setRead] = useState<Set<number>>(new Set());
  const [focus, setFocus] = useState(rememberedFocus);
  const [menuOpen, setMenuOpen] = useState(false);

  const outline = useMemo(() => readSections(lesson.body), [lesson.body]);
  const terms = useMemo(() => readTerms(lesson.body).slice(0, 8), [lesson.body]);
  const starter = lesson.exercise_detail?.starter ?? "";
  const symbols = useMemo(() => readSymbols(starter), [starter]);
  const part = splitPart(lesson.part);

  // Work out which section is on screen, how far through the lesson the reader
  // is, and which sections they have scrolled past. The section classes are
  // set on the DOM directly, so the long Markdown body never rerenders.
  const measure = useCallback(() => {
    const pane = scroller.current;
    if (!pane) return;
    const max = pane.scrollHeight - pane.clientHeight;
    setProgress(max > 0 ? Math.min(1, pane.scrollTop / max) : 1);

    const top = pane.getBoundingClientRect().top + readLine(pane);
    const nodes = Array.from(pane.querySelectorAll<HTMLElement>("section[data-section]"));
    let now = -1;
    const passed = new Set<number>();
    nodes.forEach((node, index) => {
      const box = node.getBoundingClientRect();
      if (box.top <= top) now = index;
      if (box.bottom <= top || (max > 0 && pane.scrollTop >= max - 4)) passed.add(index);
    });
    nodes.forEach((node, index) => {
      node.classList.toggle("is-current", index === now);
      node.classList.toggle("is-read", passed.has(index));
    });
    // Above the first section, the introduction is what is being read.
    pane.querySelector(".lesson-intro")?.classList.toggle("is-current", now === -1);
    setCurrent(now);
    setRead((previous) =>
      previous.size === passed.size && [...passed].every((i) => previous.has(i))
        ? previous
        : passed,
    );
  }, []);

  useEffect(() => {
    const pane = scroller.current;
    if (!pane) return;
    pane.scrollTop = 0;
    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    };
    pane.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    // KaTeX and fonts settle the layout a moment after the first paint.
    const settle = window.setTimeout(measure, 250);
    measure();
    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(settle);
      pane.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [lesson.slug, measure]);

  useEffect(() => {
    try {
      window.localStorage.setItem(FOCUS_KEY, focus ? "1" : "0");
    } catch {
      // Without storage the reader picks focus mode again next time.
    }
  }, [focus]);

  // F toggles focus mode, when the keyboard is not busy in a field or the editor.
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
      if (event.key === "f" || event.key === "F") setFocus((v) => !v);
      if (event.key === "Escape") setMenuOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function jump(id: string) {
    setMenuOpen(false);
    const target = scroller.current?.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (target?.classList.contains("term")) {
      target.classList.remove("flash");
      void target.offsetWidth;
      target.classList.add("flash");
    }
  }

  const here = current >= 0 ? outline[current] : null;

  return (
    <article
      ref={scroller}
      className="reader min-h-0 flex-1 overflow-y-auto overscroll-contain bg-paper"
      data-focus={focus ? "" : undefined}
    >
      <div className="sticky top-0 z-20 border-b border-line bg-[color-mix(in_oklab,var(--paper)_90%,transparent)] backdrop-blur-md">
        <div className="mx-auto flex h-11 max-w-[calc(var(--measure)+4rem)] items-center gap-2 px-3 sm:px-6">
          <div className="relative min-w-0 flex-1">
            <button
              type="button"
              onClick={() => setMenuOpen((v) => !v)}
              aria-expanded={menuOpen}
              aria-haspopup="menu"
              className="flex min-h-9 w-full min-w-0 items-center gap-2 rounded-[5px] px-2 text-left transition-colors hover:bg-sunk"
            >
              <span className="shrink-0 font-mono text-[10px] uppercase tracking-[0.1em] tabular-nums text-accent-ink">
                {here ? `§${current + 1}` : "Intro"}
              </span>
              <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">
                {here ? here.title : lesson.title}
              </span>
              <Chevron open={menuOpen} />
            </button>

            {menuOpen && (
              <div className="menu fade-in absolute inset-x-0 top-[calc(100%+6px)] max-h-[min(65vh,440px)] overflow-y-auto" role="menu">
                <p className="toc-title px-2.5 pb-1 pt-2">
                  On this page · {read.size} of {outline.length} read
                </p>
                <ul>
                  {outline.map((section, index) => (
                    <li key={section.id}>
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => jump(section.id)}
                        aria-current={index === current ? "true" : undefined}
                        className="menu-item"
                      >
                        <span
                          className={`w-6 shrink-0 font-mono text-[10px] tabular-nums ${read.has(index) ? "text-forest" : "text-muted"}`}
                        >
                          {read.has(index) ? <Tick /> : `§${index + 1}`}
                        </span>
                        <span className="min-w-0 flex-1 truncate">{section.title}</span>
                        {section.kind !== "concept" && (
                          <span className="meta shrink-0 uppercase">{KIND_LABEL[section.kind]}</span>
                        )}
                      </button>
                    </li>
                  ))}
                </ul>
                <p className="mt-1 hidden border-t border-line px-2.5 pb-1 pt-2 text-[12px] leading-relaxed text-muted lg:block">
                  <span className="kbd">[</span> and <span className="kbd">]</span> move between lessons.{" "}
                  <span className="kbd">F</span> toggles focus.
                </p>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => setFocus((v) => !v)}
            aria-pressed={focus}
            title="Dim everything but the section you are reading (F)"
            className="btn btn-ghost h-8 min-h-0! px-2.5!"
          >
            <FocusIcon />
            <span className="hidden sm:inline">Focus</span>
          </button>
        </div>
        <div className="h-0.5" aria-hidden>
          <div
            className="h-full origin-left bg-accent transition-transform duration-100 ease-linear"
            style={{ transform: `scaleX(${progress})` }}
          />
        </div>
      </div>

      <div className="mx-auto max-w-[calc(var(--measure)+4rem)] px-5 pb-16 pt-8 sm:px-8 lg:pt-11">
        <nav aria-label="Breadcrumb" className="breadcrumbs mb-5!">
          <Link to="/">Contents</Link>
          <span aria-hidden>/</span>
          <span>{part.number ? `Part ${part.number}` : part.name}</span>
          {part.number && (
            <>
              <span aria-hidden>/</span>
              <span>{part.name}</span>
            </>
          )}
        </nav>

        <span className="kicker">
          Lesson {String(lesson.index).padStart(2, "0")}
        </span>
        <h1 className="mt-3 font-serif text-[clamp(32px,3.4vw,48px)] font-medium leading-[1.1] tracking-[-0.035em] text-ink text-balance">
          {lesson.title}
        </h1>
        {lesson.summary && (
          <p className="mt-4 text-[16px] leading-[1.8] text-muted">{lesson.summary}</p>
        )}
        <div className="lesson-meta mt-5">
          <span>
            Lesson {lesson.index} of {lesson.count}
          </span>
          {lesson.minutes ? <span>{lesson.minutes} min</span> : null}
          {lesson.exercise_detail && <span>Exercise</span>}
          {lesson.gpu && <span>GPU</span>}
        </div>

        {lesson.objectives.length > 0 && (
          <section className="objectives mt-7 mb-0!">
            <h2 className="objectives-title">
              By the end of this lesson you can
            </h2>
            <ul className="list-disc space-y-1">
              {lesson.objectives.map((objective, index) => (
                <li key={index} className="text-ink">
                  {objective}
                </li>
              ))}
            </ul>
            {lesson.prereqs.length > 0 && (
              <p className="mt-3.5 border-t border-line pt-3 text-[13.5px] text-muted">
                Builds on{" "}
                {lesson.prereqs.map((prereq, index) => (
                  <span key={prereq}>
                    {index > 0 && (index === lesson.prereqs.length - 1 ? " and " : ", ")}
                    <Link
                      to={`/lessons/${prereq}`}
                      className="text-forest underline decoration-1 underline-offset-[3px]"
                    >
                      {titleOf(prereq, lessons)}
                    </Link>
                  </span>
                ))}
                .
              </p>
            )}
          </section>
        )}

        <div className="mt-9">
          <Markdown symbols={symbols}>{lesson.body}</Markdown>
        </div>

        {terms.length >= 3 && <Flashcards terms={terms} onFind={jump} />}

        {onStartExercise && lesson.exercise_detail && (
          <section className="mt-12 flex flex-col gap-4 rounded-[7px] border border-strong-line bg-surface p-6 sm:flex-row sm:items-center sm:gap-6">
            <div className="min-w-0 flex-1">
              <span className="kicker">Your turn</span>
              <p className="mt-1.5 font-serif text-[24px] font-medium leading-tight tracking-[-0.025em] text-ink">
                {lesson.exercise_detail.title}
              </p>
              <p className="mt-1.5 text-[14px] leading-relaxed text-muted">{lesson.exercise_detail.brief}</p>
            </div>
            <button type="button" onClick={onStartExercise} className="btn btn-primary self-start sm:self-center">
              Open the exercise
              <span aria-hidden>→</span>
            </button>
          </section>
        )}

        <nav aria-label="Other lessons" className="lesson-nav mt-12!">
          {lesson.prev && (
            <Link to={`/lessons/${lesson.prev}`} rel="prev">
              <small>Previous</small>
              {titleOf(lesson.prev, lessons)}
            </Link>
          )}
          {lesson.next && (
            <Link to={`/lessons/${lesson.next}`} rel="next">
              <small>Next</small>
              {titleOf(lesson.next, lessons)}
            </Link>
          )}
        </nav>
      </div>
    </article>
  );
}

/**
 * The lesson's defined terms as cards: the term on the front, the sentence
 * that defines it on the back. Recalling before flipping is what makes a term
 * stick, which rereading does not.
 */
function Flashcards({
  terms,
  onFind,
}: {
  terms: { id: string; term: string; sentence: string }[];
  onFind: (id: string) => void;
}) {
  const [flipped, setFlipped] = useState<Set<string>>(new Set());

  function flip(id: string) {
    setFlipped((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <section className="mt-14 border-t border-line pt-8" aria-labelledby="test-yourself">
      <span className="kicker mb-1.5">Knowledge check</span>
      <div className="mb-4 flex items-baseline gap-3">
        <h2 id="test-yourself" className="min-w-0 flex-1 font-serif text-[29px] font-medium tracking-[-0.035em] text-ink">
          Test yourself
        </h2>
        <span className="meta tabular-nums">
          {flipped.size} of {terms.length} checked
        </span>
      </div>
      <p className="-mt-2 mb-4 text-[14px] text-muted">
        Say what each term means out loud, then turn the card over to check.
      </p>
      <ul className="grid gap-2.5 sm:grid-cols-2">
        {terms.map((term) => {
          const open = flipped.has(term.id);
          return (
            <li key={term.id} className={`card ${open ? "is-flipped" : ""}`}>
              <button
                type="button"
                className="card-face"
                aria-pressed={open}
                onClick={() => flip(term.id)}
              >
                {open ? (
                  <span className="card-back">
                    <Markdown inline>{term.sentence}</Markdown>
                  </span>
                ) : (
                  <span className="flex h-full flex-col justify-between gap-3">
                    <span className="font-serif text-[19px] font-medium tracking-[-0.02em] text-ink">{term.term}</span>
                    <span className="meta">Show the definition</span>
                  </span>
                )}
              </button>
              {open && (
                <button
                  type="button"
                  className="mx-4 mb-3 self-start text-[12.5px] text-forest underline decoration-1 underline-offset-[3px]"
                  onClick={() => onFind(term.id)}
                >
                  Find it in the lesson
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      aria-hidden
      className={`shrink-0 text-muted transition-transform ${open ? "rotate-180" : ""}`}
    >
      <path d="M3 4.5 6 7.5 9 4.5" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function Tick() {
  return (
    <svg width="11" height="11" viewBox="0 0 10 10" aria-label="Read">
      <path d="M2 5.2 4.1 7.3 8 3" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FocusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path d="M1.5 4.5v-3h3M9.5 1.5h3v3M12.5 9.5v3h-3M4.5 12.5h-3v-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="7" cy="7" r="1.8" fill="currentColor" />
    </svg>
  );
}
