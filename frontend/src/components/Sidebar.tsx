import { useEffect, useMemo, useState } from "react";
import { NavLink, useParams } from "react-router-dom";

import type { Part, ProgressMap } from "../lib/api";
import { splitPart } from "../lib/parts";
import StatusMark from "./StatusMark";

type Props = {
  parts: Part[];
  progress: ProgressMap;
  /** The signed-in reader, named in the progress footer. */
  username: string;
  /** Whether the narrow-screen drawer is showing. */
  open: boolean;
  /** Whether the wide-screen reader has hidden the list. */
  collapsed: boolean;
  onClose: () => void;
  onCollapse: () => void;
  /** Opens the "/" search dialog. */
  onSearch: () => void;
};

const OPEN_PARTS_KEY = "speech.sidebar.parts";

/**
 * The lesson rail: a fixed 252px column of parts and lessons, with the
 * reader's progress at the foot. A wide screen can fold it away, and a narrow
 * one shows it as a drawer.
 */
export default function Sidebar({
  parts,
  progress,
  username,
  open,
  collapsed,
  onClose,
  onCollapse,
  onSearch,
}: Props) {
  const { slug } = useParams();
  const [shutParts, setShutParts] = useState<Set<string>>(() => {
    try {
      const raw = window.localStorage.getItem(OPEN_PARTS_KEY);
      return new Set<string>(raw ? JSON.parse(raw) : []);
    } catch {
      return new Set<string>();
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(OPEN_PARTS_KEY, JSON.stringify([...shutParts]));
    } catch {
      // Storage being off only costs the reader their collapse state.
    }
  }, [shutParts]);

  const numbers = useMemo(() => {
    const map = new Map<string, number>();
    parts.flatMap((part) => part.lessons).forEach((lesson, i) => map.set(lesson.slug, i + 1));
    return map;
  }, [parts]);

  const totals = useMemo(() => {
    const all = parts.flatMap((part) => part.lessons);
    const solved = all.filter((l) => progress[l.slug]?.status === "solved").length;
    return { all: all.length, solved };
  }, [parts, progress]);
  const share = totals.all ? (totals.solved / totals.all) * 100 : 0;

  function toggle(part: string) {
    setShutParts((current) => {
      const next = new Set(current);
      if (next.has(part)) next.delete(part);
      else next.add(part);
      return next;
    });
  }

  // A hidden rail keeps its 252px of content and loses its width, so the list
  // is clipped on the way out instead of reflowing into a narrower column.
  // visibility:hidden also takes the links out of the tab order.
  const drawer = open ? "visible translate-x-0 shadow-[var(--shadow)]" : "invisible -translate-x-full";
  const wide = collapsed
    ? "lg:invisible lg:w-0 lg:translate-x-0 lg:border-r-0"
    : "lg:visible lg:w-[var(--rail-width)] lg:translate-x-0 lg:shadow-none";

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close the lesson list"
          onClick={onClose}
          className="fixed inset-0 z-40 bg-[var(--scrim)] lg:hidden"
        />
      )}

      <nav
        aria-label="Lessons"
        className={`fixed inset-y-0 left-0 z-50 flex w-[var(--rail-width)] shrink-0 flex-col overflow-hidden border-r border-line bg-paper transition-[transform,width,visibility] duration-200 ease-out lg:static ${drawer} ${wide}`}
      >
        <div className="flex min-h-0 w-[var(--rail-width)] flex-1 flex-col">
          <div className="px-4 pb-3 pt-4">
            <div className="mb-3 flex items-center gap-2">
              <span className="kicker flex-1">Contents</span>
              <button
                type="button"
                onClick={onCollapse}
                aria-label="Hide the lesson list"
                title="Hide the lesson list (Cmd or Ctrl with B)"
                className="icon-btn -mr-1.5 hidden size-7! lg:inline-flex"
              >
                <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
                  <path d="M7.5 2.5 4 6l3.5 3.5" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {/* The drawer covers the header, and with it the button that
                  opened the drawer, so it carries its own way out. */}
              <button
                type="button"
                onClick={onClose}
                aria-label="Close the lesson list"
                className="icon-btn -mr-2 lg:hidden"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
                  <path d="M3.5 3.5l7 7M10.5 3.5l-7 7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                </svg>
              </button>
            </div>

            <button type="button" onClick={onSearch} className="search-trigger w-full">
              <SearchIcon />
              <span className="flex-1 text-left">Search lessons</span>
              <kbd className="key">/</kbd>
            </button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2.5 pb-4">
            {parts.map((part) => {
              const shut = shutParts.has(part.part);
              const { number, name } = splitPart(part.part);
              const done = part.lessons.filter((l) => progress[l.slug]?.status === "solved").length;
              const complete = part.lessons.length > 0 && done === part.lessons.length;
              return (
                <section key={part.part} className="mt-3 first:mt-1">
                  <button
                    type="button"
                    onClick={() => toggle(part.part)}
                    aria-expanded={!shut}
                    className="sidebar-group-title group m-0! flex w-full items-baseline gap-2 rounded-[5px] px-2 py-1.5 text-left transition-colors hover:bg-sunk hover:text-ink"
                  >
                    <span className="min-w-0 flex-1">
                      {number && <span className="text-accent-ink">{number.padStart(2, "0")} · </span>}
                      {name}
                    </span>
                    <span className={`tabular-nums ${complete ? "text-forest" : ""}`}>
                      {done}/{part.lessons.length}
                    </span>
                    <svg
                      width="10"
                      height="10"
                      viewBox="0 0 10 10"
                      className={`shrink-0 self-center transition-transform duration-150 ${shut ? "-rotate-90" : ""}`}
                      aria-hidden
                    >
                      <path d="M2.5 3.75 5 6.25l2.5-2.5" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>

                  {!shut && (
                    <ul className="mt-0.5 space-y-px">
                      {part.lessons.map((lesson) => {
                        const state = progress[lesson.slug]?.status;
                        const current = slug === lesson.slug;
                        return (
                          <li key={lesson.slug}>
                            <NavLink
                              to={`/lessons/${lesson.slug}`}
                              onClick={onClose}
                              aria-current={current ? "page" : undefined}
                              className={`sidebar-lesson max-lg:py-2 ${state === "solved" ? "is-complete" : ""}`}
                            >
                              <span className="sidebar-lesson-number tabular-nums">
                                {numbers.get(lesson.slug)}
                              </span>
                              <span className="min-w-0 flex-1">{lesson.title}</span>
                              <StatusMark state={state} className="translate-y-[2px] self-start" />
                            </NavLink>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </section>
              );
            })}
          </div>

          <footer className="shrink-0 border-t border-line px-4 pb-4 pt-3.5">
            <div className="mb-2 flex items-baseline justify-between gap-2 text-[12px]">
              <span className="font-semibold text-ink">Your progress</span>
              <span className="meta tabular-nums">
                {totals.solved} / {totals.all}
              </span>
            </div>
            <div
              className="progress-track"
              role="progressbar"
              aria-label="Lessons solved"
              aria-valuemin={0}
              aria-valuemax={totals.all}
              aria-valuenow={totals.solved}
            >
              <div className="progress-fill" style={{ width: `${share}%` }} />
            </div>
            <p className="meta mt-2.5 flex items-center gap-1.5">
              <LockIcon />
              <span className="min-w-0 truncate">Saved for {username}</span>
            </p>
          </footer>
        </div>
      </nav>
    </>
  );
}

function SearchIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 12 12" aria-hidden className="shrink-0">
      <circle cx="5" cy="5" r="3.4" stroke="currentColor" strokeWidth="1.2" fill="none" />
      <path d="M7.6 7.6 10.5 10.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}

function LockIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden className="shrink-0">
      <rect x="1.75" y="4.25" width="6.5" height="4.75" rx="1" stroke="currentColor" strokeWidth="1" fill="none" />
      <path d="M3.25 4.25V3a1.75 1.75 0 0 1 3.5 0v1.25" stroke="currentColor" strokeWidth="1" fill="none" />
    </svg>
  );
}
