import { Fragment, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";

import type { LessonMeta, Part } from "../lib/api";
import { lessonNumbers, splitPart } from "../lib/parts";

type Props = {
  open: boolean;
  onClose: () => void;
  /** Every part, projects included. The search reads the lesson index only. */
  parts: Part[];
};

type Entry = {
  lesson: LessonMeta;
  number: string;
  part: string;
  /** Lowercased text the query is matched against. */
  title: string;
  rest: string;
};

type Hit = { entry: Entry; score: number; detail: string };

const LIMIT = 30;

/**
 * The "/" search dialog: lesson titles, summaries, objectives, and part names
 * from the /api/lessons index, which the app has already loaded. Arrow keys
 * move through the results, Return opens one, and Escape closes the dialog.
 */
export default function SearchDialog({ open, onClose, parts }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);

  const entries = useMemo<Entry[]>(() => {
    const all = parts.flatMap((part) => part.lessons);
    const numbers = lessonNumbers(all);
    return parts.flatMap((part) =>
      part.lessons.map((lesson) => ({
        lesson,
        number: numbers.get(lesson.slug) ?? "",
        part: splitPart(part.part).name,
        title: lesson.title.toLowerCase(),
        rest: [lesson.summary, ...lesson.objectives, part.part, lesson.slug].join("\n").toLowerCase(),
      })),
    );
  }, [parts]);

  const words = useMemo(
    () => query.toLowerCase().split(/\s+/).filter((word) => word.length > 0),
    [query],
  );

  const hits = useMemo<Hit[]>(() => {
    if (words.length === 0) return [];
    const found: Hit[] = [];
    for (const entry of entries) {
      if (!words.every((word) => entry.title.includes(word) || entry.rest.includes(word))) continue;
      let score = 0;
      for (const word of words) {
        if (entry.title.startsWith(word)) score += 6;
        else if (new RegExp(`\\b${escape(word)}`).test(entry.title)) score += 4;
        else if (entry.title.includes(word)) score += 2;
        else score += 1;
      }
      found.push({ entry, score, detail: detailFor(entry.lesson, words) });
    }
    return found.sort((a, b) => b.score - a.score).slice(0, LIMIT);
  }, [entries, words]);

  useEffect(() => setSelected(0), [query]);

  useEffect(() => {
    const box = dialog.current;
    if (!box) return;
    if (open && !box.open) {
      setQuery("");
      box.showModal();
      input.current?.focus();
    } else if (!open && box.open) {
      box.close();
    }
  }, [open]);

  // Keep the selected result in view as the arrow keys move through a long list.
  useEffect(() => {
    dialog.current
      ?.querySelector(`[data-index="${selected}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  function go(hit: Hit | undefined) {
    if (!hit) return;
    onClose();
    navigate(`/lessons/${hit.entry.lesson.slug}`);
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelected((i) => Math.min(i + 1, Math.max(hits.length - 1, 0)));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelected((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      go(hits[selected]);
    }
  }

  return (
    <dialog
      ref={dialog}
      className="search-dialog"
      aria-label="Search the lessons"
      onClose={onClose}
      onClick={(event) => {
        // A click on the backdrop lands on the dialog element itself.
        if (event.target === dialog.current) onClose();
      }}
    >
      <input
        ref={input}
        type="text"
        enterKeyHint="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Search lessons, summaries, and objectives"
        aria-label="Search the lessons"
        aria-controls="search-results"
        aria-activedescendant={hits[selected] ? `search-hit-${selected}` : undefined}
        autoComplete="off"
        spellCheck={false}
      />

      {words.length === 0 ? (
        <p className="px-[18px] py-4 text-[13px] leading-relaxed text-muted">
          Type to search {entries.length} lessons by title, summary, or objective.{" "}
          <span className="hidden sm:inline">
            Use <kbd className="key">↑</kbd> <kbd className="key">↓</kbd> to move and{" "}
            <kbd className="key">Return</kbd> to open.
          </span>
        </p>
      ) : hits.length === 0 ? (
        <p className="px-[18px] py-4 text-[13px] text-muted">No lesson matches “{query.trim()}”.</p>
      ) : (
        <ul id="search-results" role="listbox" aria-label="Results" className="search-results">
          {hits.map((hit, index) => (
            <li key={hit.entry.lesson.slug}>
              <a
                id={`search-hit-${index}`}
                data-index={index}
                role="option"
                aria-selected={index === selected}
                href={`/lessons/${hit.entry.lesson.slug}`}
                className="search-result"
                onMouseMove={() => setSelected(index)}
                onClick={(event) => {
                  event.preventDefault();
                  go(hit);
                }}
              >
                <span className="kicker mb-0.5">
                  Lesson {hit.entry.number} ·{" "}
                  {hit.entry.part}
                </span>
                <span className="search-result-title">{mark(hit.entry.lesson.title, words)}</span>
                {hit.detail && (
                  <span className="search-result-detail line-clamp-2">{mark(hit.detail, words)}</span>
                )}
              </a>
            </li>
          ))}
        </ul>
      )}
    </dialog>
  );
}

/** The objective that matches the most words, or else the summary. */
function detailFor(lesson: LessonMeta, words: string[]): string {
  let best = lesson.summary;
  let bestCount = count(lesson.summary, words);
  for (const objective of lesson.objectives) {
    const n = count(objective, words);
    if (n > bestCount) {
      best = objective;
      bestCount = n;
    }
  }
  return best;
}

function count(text: string, words: string[]): number {
  const lower = text.toLowerCase();
  return words.filter((word) => lower.includes(word)).length;
}

function escape(word: string): string {
  return word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The text with every query word wrapped in <mark>. */
function mark(text: string, words: string[]): ReactNode {
  if (words.length === 0) return text;
  const pattern = new RegExp(`(${words.map(escape).join("|")})`, "gi");
  return text.split(pattern).map((piece, index) =>
    index % 2 === 1 ? <mark key={index}>{piece}</mark> : <Fragment key={index}>{piece}</Fragment>,
  );
}
