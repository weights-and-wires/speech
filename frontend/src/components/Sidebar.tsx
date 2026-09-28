import { useEffect, useMemo, useState } from "react";
import { NavLink, useParams } from "react-router-dom";

import type { Part, ProgressMap } from "../lib/api";
import { splitPart } from "../lib/parts";
import StatusMark from "./StatusMark";

type Props = {
  parts: Part[];
  progress: ProgressMap;
  /** Whether the narrow-screen drawer is showing. */
  open: boolean;
  /** Whether the wide-screen reader has hidden the list. */
  collapsed: boolean;
  onClose: () => void;
  onCollapse: () => void;
};

const OPEN_PARTS_KEY = "speech.sidebar.parts";

export default function Sidebar({
  parts,
  progress,
  open,
  collapsed,
  onClose,
  onCollapse,
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
  const [filter, setFilter] = useState("");

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

  const query = filter.trim().toLowerCase();
  const shown = useMemo(() => {
    if (!query) return parts;
    return parts
      .map((part) => ({
        ...part,
        lessons: part.lessons.filter(
          (lesson) =>
            lesson.title.toLowerCase().includes(query) ||
            lesson.summary.toLowerCase().includes(query) ||
            lesson.slug.includes(query),
        ),
      }))
      .filter((part) => part.lessons.length > 0);
  }, [parts, query]);

  const totals = useMemo(() => {
    const all = parts.flatMap((part) => part.lessons);
    const solved = all.filter((l) => progress[l.slug]?.status === "solved").length;
    return { all: all.length, solved };
  }, [parts, progress]);

  function toggle(part: string) {
    setShutParts((current) => {
      const next = new Set(current);
      if (next.has(part)) next.delete(part);
      else next.add(part);
      return next;
    });
  }

  // A hidden pane keeps its 18rem of content and loses its width, so the list
  // is clipped on the way out instead of reflowing into a narrower column.
  // visibility:hidden also takes the links out of the tab order.
  const drawer = open ? "visible translate-x-0 shadow-2xl" : "invisible -translate-x-full";
  const wide = collapsed
    ? "lg:invisible lg:w-0 lg:translate-x-0 lg:border-r-0"
    : "lg:visible lg:w-72 lg:translate-x-0";

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close the lesson list"
          onClick={onClose}
          className="fixed inset-0 z-40 bg-black/30 lg:hidden"
        />
      )}

      <nav
        aria-label="Lessons"
        className={`fixed inset-y-0 left-0 z-50 flex w-72 shrink-0 flex-col overflow-hidden border-r border-rule bg-chrome transition-[transform,width,visibility] duration-200 ease-out lg:static ${drawer} ${wide}`}
      >
        <div className="flex min-h-0 w-72 flex-1 flex-col">
          <div className="px-4 pb-3 pt-3.5">
            <div className="mb-2 flex items-center gap-2">
              <span className="flex-1 text-[13px] font-semibold text-ink">Contents</span>
              <span className="text-[12px] tabular-nums text-ink-3">
                {totals.solved} of {totals.all} solved
              </span>
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
            <div className="h-[3px] overflow-hidden rounded-full bg-rule">
              <div
                className="h-full rounded-full bg-good transition-[width] duration-500"
                style={{ width: `${totals.all ? (totals.solved / totals.all) * 100 : 0}%` }}
              />
            </div>

            <div className="relative mt-3">
              <svg
                width="13"
                height="13"
                viewBox="0 0 12 12"
                aria-hidden
                className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-4"
              >
                <circle cx="5" cy="5" r="3.4" stroke="currentColor" strokeWidth="1.2" fill="none" />
                <path d="M7.6 7.6 10.5 10.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
              <input
                type="search"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") setFilter("");
                }}
                placeholder="Find a lesson"
                aria-label="Find a lesson"
                className="field py-1.5! pl-8!"
              />
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2 pb-4">
            {shown.map((part) => {
              const shut = shutParts.has(part.part) && !query;
              const { number, name } = splitPart(part.part);
              const done = part.lessons.filter((l) => progress[l.slug]?.status === "solved").length;
              const complete = part.lessons.length > 0 && done === part.lessons.length;
              return (
                <section key={part.part} className="mt-2">
                  <button
                    type="button"
                    onClick={() => toggle(part.part)}
                    aria-expanded={!shut}
                    className="group flex w-full items-baseline gap-2 rounded-md px-2 py-2 text-left transition-colors hover:bg-[color-mix(in_oklab,var(--ink)_5%,transparent)] lg:py-1.5"
                  >
                    <span className="w-5 shrink-0 text-[11.5px] tabular-nums text-ink-4">
                      {number ? `${number}.` : ""}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-ink-2">
                      {name}
                    </span>
                    <span className={`text-[11.5px] tabular-nums ${complete ? "text-good" : "text-ink-4"}`}>
                      {done}/{part.lessons.length}
                    </span>
                    <svg
                      width="10"
                      height="10"
                      viewBox="0 0 10 10"
                      className={`shrink-0 self-center text-ink-4 transition-transform duration-150 ${shut ? "-rotate-90" : ""}`}
                      aria-hidden
                    >
                      <path d="M2.5 3.75 5 6.25l2.5-2.5" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>

                  {!shut && (
                    <ul className="mt-px space-y-px">
                      {part.lessons.map((lesson) => {
                        const state = progress[lesson.slug]?.status;
                        const current = slug === lesson.slug;
                        return (
                          <li key={lesson.slug}>
                            <NavLink
                              to={`/lessons/${lesson.slug}`}
                              onClick={onClose}
                              aria-current={current ? "page" : undefined}
                              className={`flex items-start gap-2 rounded-md px-2 py-2 text-[13px] leading-snug transition-colors lg:py-[5px] ${
                                current
                                  ? "bg-raise font-medium text-ink shadow-[inset_0_0_0_1px_var(--rule)]"
                                  : "text-ink-2 hover:bg-[color-mix(in_oklab,var(--ink)_5%,transparent)] hover:text-ink"
                              }`}
                            >
                              <span className="w-5 shrink-0 pt-px text-[11.5px] tabular-nums text-ink-4">
                                {numbers.get(lesson.slug)}
                              </span>
                              <span className="min-w-0 flex-1">{lesson.title}</span>
                              <StatusMark state={state} className="mt-[3px]" />
                            </NavLink>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </section>
              );
            })}

            {shown.length === 0 && (
              <p className="px-3 py-8 text-center text-[13px] text-ink-3">
                No lesson matches “{filter.trim()}”.
              </p>
            )}
          </div>
        </div>
      </nav>
    </>
  );
}
