/**
 * The PDF book: every lesson and exercise on one long page, laid out for print.
 *
 * scripts/build-book.mjs opens this page in headless Chrome, waits for
 * `window.__book.ready`, prints it once to find out where everything landed,
 * fills the page numbers in with `window.__setPages`, and prints it again.
 * The lesson bodies go through the same Markdown component the site uses, so
 * the book and the site cannot drift apart, and the demos render as still
 * figures: each one inside `PrintedFigure`, drawn in its starting state or in
 * the state `useWhenPrinted` sets up, with its controls hidden and a line
 * naming the lesson online where the reader can use it.
 *
 * Each lesson opens on a page of its own with what it promises and its
 * contents, runs through its introduction and sections, and ends
 * with its exercise: the brief, the starter file, and the hints. The worked
 * solutions sit together in an appendix, so they are never on the page
 * facing the exercise they give away.
 */

import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import { useEffect, useLayoutEffect, useRef, type ReactNode } from "react";

import Markdown from "../components/Markdown";
import { PrintedFigure } from "../demos/kit";
import Mark from "../components/Mark";
import type { ExerciseDetail, LessonMeta } from "../lib/api";
import { KIND_LABEL, sections } from "../lib/lessonOutline";
import { splitPart } from "../lib/parts";

hljs.registerLanguage("python", python);

export const SITE = "speech.ifkash.dev";
const TITLE = "Speech and audio";

export type BookLesson = LessonMeta & { body: string };
export type BookExercise = ExerciseDetail & { solution: string };
export type BookData = { commit: string; lessons: BookLesson[]; exercises: Record<string, BookExercise> };

/** One bookmark in the PDF's outline. `id` is the anchor it points at. */
export type OutlineEntry = { id: string; title: string; children: OutlineEntry[] };

export type BookReport = {
  error?: string;
  ready?: boolean;
  title?: string;
  subject?: string;
  outline?: OutlineEntry[];
  katexErrors?: string[];
  overflow?: string[];
  figures?: number;
};

declare global {
  interface Window {
    __book?: BookReport;
    __setPages?: (pages: Record<string, number>) => void;
  }
}

const lessonId = (slug: string) => `ls-${slug}`;
const bodyId = (slug: string) => `ls-${slug}--body`;
const exerciseId = (slug: string) => `ex-${slug}`;
const solutionId = (slug: string) => `sol-${slug}`;
const pageName = (slug: string) => `pg-${slug}`;

type Numbered = { lesson: BookLesson; n: number; exercise?: BookExercise };

/** A CSS string literal, for the running heads. */
function cssString(text: string): string {
  return `"${text.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, " ")}"`;
}

/** Heading text without its Markdown: backticks, emphasis, and dollar signs. */
function plain(text: string): string {
  return text.replace(/[`*_]/g, "").replace(/\$([^$]*)\$/g, "$1");
}

/** Lessons grouped into parts, in the order the parts first appear. */
function groupParts(lessons: Numbered[]) {
  const parts: { part: string; number: string; name: string; lessons: Numbered[] }[] = [];
  for (const entry of lessons) {
    let last = parts[parts.length - 1];
    if (!last || last.part !== entry.lesson.part) {
      const { number, name } = splitPart(entry.lesson.part);
      last = { part: entry.lesson.part, number: number || String(parts.length + 1), name, lessons: [] };
      parts.push(last);
    }
    last.lessons.push(entry);
  }
  return parts;
}

/** Waits for the lazy demos, the fonts, and the layout to settle. */
async function settled(): Promise<void> {
  // A demo shows a busy placeholder until its chunk has loaded.
  const deadline = Date.now() + 90_000;
  let quiet = 0;
  while (Date.now() < deadline && quiet < 5) {
    await new Promise((resolve) => window.setTimeout(resolve, 100));
    quiet = document.querySelector('[aria-busy="true"]') ? 0 : quiet + 1;
  }
  await document.fonts.ready;
  await new Promise((resolve) => window.setTimeout(resolve, 300));
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

/**
 * Print has no horizontal scroll, so anything wider than the text block is
 * shrunk until it fits, and whatever still does not fit is reported.
 */
function fitToMeasure(): string[] {
  const overflow: string[] = [];
  // `box` takes the smaller type, `inner` is what has to fit, and `width` is
  // the room it has.
  const shrink = (box: HTMLElement, inner: HTMLElement, width: number, what: string, floor: number) => {
    let scale = 1;
    for (let step = 0; step < 8 && inner.scrollWidth > width + 1 && scale > floor; step += 1) {
      scale = Math.max(floor, scale * Math.min(0.97, width / inner.scrollWidth));
      box.style.fontSize = `${scale}em`;
    }
    if (inner.scrollWidth > width + 1) {
      const where = box.closest("[data-where]")?.getAttribute("data-where") ?? "?";
      overflow.push(`${where}: ${what}: ${(box.textContent ?? "").slice(0, 70)}`);
    }
  };
  document.querySelectorAll<HTMLElement>(".katex-display").forEach((display) => {
    const katex = display.querySelector<HTMLElement>(".katex");
    if (katex) shrink(display, katex, display.clientWidth, "display maths", 0.55);
  });
  document.querySelectorAll<HTMLElement>(".lesson table").forEach((table) => {
    const room = table.parentElement?.clientWidth ?? table.clientWidth;
    shrink(table, table, room, "table", 0.7);
  });
  return overflow;
}

export default function Book({ data }: { data: BookData }) {
  const numbered: Numbered[] = data.lessons.map((lesson, index) => ({
    lesson,
    n: index + 1,
    exercise: lesson.exercise ? data.exercises[lesson.exercise] : undefined,
  }));
  const bySlug = new Map(numbered.map((entry) => [entry.lesson.slug, entry]));
  const parts = groupParts(numbered);
  const withSolutions = numbered.filter((entry) => entry.exercise?.solution);

  useEffect(() => {
    document.title = TITLE;

    window.__setPages = (pages) => {
      document.querySelectorAll<HTMLElement>("[data-pg]").forEach((node) => {
        const page = pages[node.dataset.pg ?? ""];
        node.textContent = page ? String(page) : "";
      });
    };

    const outline: OutlineEntry[] = [
      { id: "contents", title: "Contents", children: [] },
      { id: "about", title: "How to use this book", children: [] },
      ...parts.map((part) => ({
        id: `part-${part.number}`,
        title: `Part ${part.number}. ${part.name}`,
        children: part.lessons.map(({ lesson, n, exercise }) => ({
          id: lessonId(lesson.slug),
          title: `${n}. ${lesson.title}`,
          children: [
            ...sections(lesson.body).map((section, i) => ({
              id: `${bodyId(lesson.slug)}--${section.id}`,
              title: `§${i + 1} ${plain(section.title)}`,
              children: [],
            })),
            ...(exercise ? [{ id: exerciseId(lesson.slug), title: `Exercise ${n}`, children: [] }] : []),
          ],
        })),
      })),
      {
        id: "solutions",
        title: "Appendix. Solutions",
        children: withSolutions.map(({ lesson, n }) => ({
          id: solutionId(lesson.slug),
          title: `${n}. ${lesson.title}`,
          children: [],
        })),
      },
    ];

    let live = true;
    settled().then(() => {
      if (!live) return;
      const overflow = fitToMeasure();
      window.__book = {
        ready: true,
        title: TITLE,
        subject: `A hands-on course in speech and audio, in ${numbered.length} lessons, with every exercise and its solution.`,
        outline,
        katexErrors: Array.from(document.querySelectorAll<HTMLElement>(".katex-error")).map(
          (node) => `${node.closest("[data-where]")?.getAttribute("data-where") ?? "?"}: ${node.title || node.textContent}`,
        ),
        overflow,
        figures: document.querySelectorAll("figure.demo").length,
      };
    });
    return () => {
      live = false;
    };
    // The book renders once per page load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The running heads: the book's title on the left-hand page, the lesson on
  // the right. Each lesson is its own named page so its right-hand head can
  // say which lesson it is.
  const pageRules = numbered
    .map(({ lesson, n }) => {
      const name = pageName(lesson.slug);
      return `.${name}{page:${name}}@page ${name}:right{@top-right{content:${cssString(`${n} · ${lesson.title}`)}}}`;
    })
    .join("\n");
  // Chrome lets a later :left rule beat an earlier named page, whatever the
  // specificity, so the pages without a running head say so again here, one
  // rule each, because Chrome ignores a list of page names.
  const heads = [
    `@page :left{@top-left{content:${cssString(TITLE)}}}`,
    `@page solutions:right{@top-right{content:"Appendix · Solutions"}}`,
    ...["cover", "front", "part", "opener"].map((name) => `@page ${name}{@top-left{content:none}@top-right{content:none}}`),
  ].join("\n");

  return (
    <div className="book">
      <style>{`${heads}\n${pageRules}`}</style>

      <Cover parts={parts} lessons={numbered.length} commit={data.commit} />
      <Contents parts={parts} solutions={withSolutions.length > 0} />
      <About lessons={numbered.length} />

      {parts.map((part) => (
        <div key={part.part}>
          <section id={`part-${part.number}`} className="part-page">
            <p className="part-kicker">Part {part.number}</p>
            <h1 className="part-title">{part.name}</h1>
            <ol className="part-chapters">
              {part.lessons.map(({ lesson, n }) => (
                <li key={lesson.slug}>
                  <span className="part-chapter-n">{n}</span>
                  <span className="min-w-0 flex-1">
                    <span className="part-chapter-title">{lesson.title}</span>
                    {lesson.summary && <span className="part-chapter-q">{lesson.summary}</span>}
                  </span>
                  <a href={`#${lessonId(lesson.slug)}`} className="part-chapter-pg">
                    <span className="pg" data-pg={lessonId(lesson.slug)} />
                  </a>
                </li>
              ))}
            </ol>
          </section>

          {part.lessons.map((entry) => (
            <Lesson key={entry.lesson.slug} entry={entry} bySlug={bySlug} />
          ))}
        </div>
      ))}

      {withSolutions.length > 0 && <Solutions entries={withSolutions} />}
    </div>
  );
}

function Cover({
  parts,
  lessons,
  commit,
}: {
  parts: ReturnType<typeof groupParts>;
  lessons: number;
  commit: string;
}) {
  const built = new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  return (
    <section className="cover">
      <div className="cover-top">
        <Mark size={44} />
        <span className="cover-site">{SITE}</span>
      </div>
      <div className="cover-main">
        <p className="cover-kicker">{lessons} lessons, each with an exercise to build</p>
        <h1 className="cover-title">{TITLE}</h1>
        <p className="cover-subtitle">A hands-on course in how machines hear, speak, and process sound</p>
        <div className="cover-edition">
          <p className="cover-edition-name">The complete course</p>
          <p className="cover-edition-blurb">
            Every lesson, every exercise with its starter code and hints, and the worked solutions at the back.
          </p>
        </div>
      </div>
      <div className="cover-foot">
        <ol className="cover-parts">
          {parts.map((part) => (
            <li key={part.part}>
              <span className="cover-parts-n">{part.number}</span>
              {part.name}
            </li>
          ))}
        </ol>
        <p className="cover-built">
          Built {built}
          {commit && commit !== "unknown" ? ` from ${commit.slice(0, 7)}` : ""}
        </p>
      </div>
    </section>
  );
}

function Contents({ parts, solutions }: { parts: ReturnType<typeof groupParts>; solutions: boolean }) {
  return (
    <nav id="contents" className="front toc" aria-label="Contents">
      <h1 className="front-title">Contents</h1>
      <a href="#about" className="toc-row toc-front">
        <span className="toc-title">How to use this book</span>
        <span className="toc-leader" />
        <span className="pg" data-pg="about" />
      </a>
      {parts.map((part) => (
        <div key={part.part} className="toc-part">
          <a href={`#part-${part.number}`} className="toc-part-title">
            <span className="toc-part-n">Part {part.number}</span>
            <span className="toc-title">{part.name}</span>
            <span className="toc-leader" />
            <span className="pg" data-pg={`part-${part.number}`} />
          </a>
          {part.lessons.map(({ lesson, n }) => (
            <div key={lesson.slug} className="toc-chapter">
              <a href={`#${lessonId(lesson.slug)}`} className="toc-row">
                <span className="toc-n">{n}</span>
                <span className="toc-title">{lesson.title}</span>
                <span className="toc-leader" />
                <span className="pg" data-pg={lessonId(lesson.slug)} />
              </a>
            </div>
          ))}
        </div>
      ))}
      {solutions && (
        <div className="toc-part">
          <a href="#solutions" className="toc-part-title">
            <span className="toc-part-n">Appendix</span>
            <span className="toc-title">Solutions to the exercises</span>
            <span className="toc-leader" />
            <span className="pg" data-pg="solutions" />
          </a>
        </div>
      )}
    </nav>
  );
}

function About({ lessons }: { lessons: number }) {
  return (
    <section id="about" className="front about">
      <h1 className="front-title">How to use this book</h1>
      <div className="lesson">
        <p>
          This course teaches speech and audio processing by having you build it. There are {lessons} lessons, and
          every one ends with an exercise: a Python file of functions to fill in, and a set of checks that test what
          your functions compute, not how your code looks.
        </p>
        <p>
          You need to write Python. Every term and symbol is defined where it first appears, and every formula sits
          next to the numpy code that computes it. The lessons online play sound and let you push on interactive
          demos; in this book each demo is a still figure, and each one names the page where you can use it.
        </p>
      </div>

      <dl className="about-list">
        <div>
          <dt>Each lesson</dt>
          <dd>
            Opens on a page of its own with what you will be able to do, the lessons to read first, and its sections
            with their page numbers. The sections run in the same order every time: what you already know, the idea,
            a worked example by hand, how to build it step by step, what it costs, and where it breaks.
          </dd>
        </div>
        <div>
          <dt>Each exercise</dt>
          <dd>
            Follows its lesson, with the brief, the whole starter file, and the hints. The starter file is the
            specification: each function's docstring says what it takes and what it must return.
          </dd>
        </div>
        <div>
          <dt>The checks</dt>
          <dd>
            Run online. Open the lesson at {SITE}, write your code in the editor beside it, and press Check. The
            checks report one at a time as they finish.
          </dd>
        </div>
        <div>
          <dt>The solutions</dt>
          <dd>
            Are in the appendix, one per exercise, and each exercise names the page its solution is on. Try the
            exercise first: the checks were written against these solutions, but any correct answer passes them.
          </dd>
        </div>
      </dl>

      <div className="lesson about-notes">
        <p>
          <strong className="runin">Boxes.</strong> A worked example to follow with pencil and paper sits in a shaded
          box marked "Try it by hand". A key idea, a tip, or a warning has a coloured edge. A displayed formula has a
          frame, with its "Read it aloud" explanation beneath it.
        </p>
        <p>
          <strong className="runin">Code.</strong> In the starter files and solutions, a line too long for the page
          carries on below, indented further than the line it belongs to. Type it as one line.
        </p>
      </div>
    </section>
  );
}

/** Prefixes every id under `root`, since every lesson uses the same headings. */
function usePrefixedIds(prefix: string) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    ref.current?.querySelectorAll<HTMLElement>("[id]").forEach((node) => {
      if (!node.id.startsWith(`${prefix}--`)) node.id = `${prefix}--${node.id}`;
    });
  }, [prefix]);
  return ref;
}

function Lesson({ entry, bySlug }: { entry: Numbered; bySlug: Map<string, Numbered> }) {
  const { lesson, n, exercise } = entry;
  const outline = sections(lesson.body);
  const part = splitPart(lesson.part);
  const bodyRef = usePrefixedIds(bodyId(lesson.slug));
  const prereqs = lesson.prereqs.map((slug) => bySlug.get(slug)).filter((found): found is Numbered => !!found);

  return (
    <article className="chapter">
      <header id={lessonId(lesson.slug)} className="opener">
        <p className="opener-kicker">
          <span className="opener-n">Lesson {n}</span>
          <span>{part.number ? `Part ${part.number}, ${part.name}` : part.name}</span>
          {lesson.minutes ? <span className="opener-minutes">About {lesson.minutes} minutes</span> : null}
        </p>
        <h1 className="opener-title">{lesson.title}</h1>
        {lesson.summary && <p className="opener-question">{lesson.summary}</p>}

        <div className="opener-grid">
          {lesson.objectives.length > 0 && (
            <div className="opener-box">
              <p className="mini-toc-label">By the end, you can</p>
              <ul className="opener-objectives">
                {lesson.objectives.map((objective) => (
                  <li key={objective}>{objective}</li>
                ))}
              </ul>
            </div>
          )}
          {prereqs.length > 0 && (
            <div className="opener-box">
              <p className="mini-toc-label">Read first</p>
              <ul className="opener-prereqs">
                {prereqs.map((found) => (
                  <li key={found.lesson.slug}>
                    <a href={`#${lessonId(found.lesson.slug)}`}>
                      <span className="opener-prereq-n">{found.n}</span>
                      <span className="toc-title">{found.lesson.title}</span>
                      <span className="toc-leader" />
                      <span className="pg" data-pg={lessonId(found.lesson.slug)} />
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {outline.length > 0 && (
          <div className="opener-levels">
            <p className="mini-toc-label">In this lesson</p>
            <ol className="mini-toc">
              {outline.map((section, index) => (
                <li key={section.id}>
                  <a href={`#${bodyId(lesson.slug)}--${section.id}`}>
                    <span className="mini-toc-n">§{index + 1}</span>
                    <span className="toc-title">{plain(section.title)}</span>
                    {section.kind !== "concept" && <span className="mini-toc-kind">{KIND_LABEL[section.kind]}</span>}
                    <span className="toc-leader" />
                    <span className="pg" data-pg={`${bodyId(lesson.slug)}--${section.id}`} />
                  </a>
                </li>
              ))}
              {exercise && (
                <li className="mini-toc-exercise">
                  <a href={`#${exerciseId(lesson.slug)}`}>
                    <span className="mini-toc-n">Ex.</span>
                    <span className="toc-title">{exercise.title}</span>
                    <span className="toc-leader" />
                    <span className="pg" data-pg={exerciseId(lesson.slug)} />
                  </a>
                </li>
              )}
            </ol>
          </div>
        )}
      </header>

      <section className={`level ${pageName(lesson.slug)}`}>
        <div ref={bodyRef} className="level-body" data-where={lesson.slug}>
          <PrintedFigure.Provider value={{ url: `${SITE}/lessons/${lesson.slug}` }}>
            <Markdown>{lesson.body}</Markdown>
          </PrintedFigure.Provider>
        </div>
        {exercise && <Exercise slug={lesson.slug} n={n} exercise={exercise} />}
      </section>
    </article>
  );
}

function Exercise({ slug, n, exercise }: { slug: string; n: number; exercise: BookExercise }) {
  return (
    <section id={exerciseId(slug)} className="exercise" data-where={`${slug} exercise`}>
      <p className="exercise-kicker">
        <span>Exercise {n}</span>
        {exercise.packages.length > 0 && <span className="exercise-packages">Uses {exercise.packages.join(", ")}</span>}
      </p>
      <h2 className="exercise-title">{exercise.title}</h2>
      {/* The site shows the brief and the hints as plain text, and so does the book. */}
      {exercise.brief && <p className="exercise-brief">{exercise.brief}</p>}
      <p className="exercise-online">
        Write it at <a href={`https://${SITE}/lessons/${slug}`}>{`${SITE}/lessons/${slug}`}</a>, where the checks run
        beside the editor.
        {exercise.solution && (
          <>
            {" "}
            The solution is on{" "}
            <a href={`#${solutionId(slug)}`}>
              page <span className="pg pg-inline" data-pg={solutionId(slug)} />
            </a>
            .
          </>
        )}
      </p>

      <CodeFile name="starter.py" code={exercise.starter} />

      {exercise.hints.length > 0 && (
        <div className="exercise-hints">
          <h3 className="check-title">Hints</h3>
          <p className="exercise-hints-note">Read one at a time, and only when you are stuck.</p>
          <ol className="check-list">
            {exercise.hints.map((hint, index) => (
              <li key={index}>{hint}</li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function Solutions({ entries }: { entries: Numbered[] }) {
  return (
    <div className="solutions">
      <section id="solutions" className="part-page">
        <p className="part-kicker">Appendix</p>
        <h1 className="part-title">Solutions to the exercises</h1>
        <div className="lesson solutions-note">
          <p>
            These are the reference solutions the checks were written against. Each one is a correct answer, not the
            only one: any code that passes the checks is right. Try the exercise first, and read a solution after your
            own code passes, or when the hints have not been enough.
          </p>
        </div>
      </section>
      <div className="solutions-body">
        {entries.map(({ lesson, n, exercise }) => (
          <section key={lesson.slug} id={solutionId(lesson.slug)} className="solution" data-where={`${lesson.slug} solution`}>
            <p className="exercise-kicker">
              <span>Solution {n}</span>
              <a href={`#${exerciseId(lesson.slug)}`} className="solution-back">
                Exercise on page <span className="pg pg-inline" data-pg={exerciseId(lesson.slug)} />
              </a>
            </p>
            <h2 className="exercise-title">{exercise!.title}</h2>
            <p className="solution-lesson">For lesson {n}, {lesson.title}</p>
            <CodeFile name="solution.py" code={exercise!.solution} />
          </section>
        ))}
      </div>
    </div>
  );
}

/**
 * Highlighted HTML split into one block per source line. A token that spans
 * lines, such as a docstring, is closed at the end of each line and opened
 * again on the next, so every line is balanced markup on its own.
 *
 * Each line carries its indentation, so a line too long for the page wraps
 * with a hanging indent: the rest of it sits further in than the line it
 * continues, and Python copied from the page keeps its structure.
 */
function splitLines(html: string, source: string): string {
  const sourceLines = source.split("\n");
  const open: string[] = [];
  const lines: string[] = [];
  let current = "";
  for (const token of html.split(/(<span[^>]*>|<\/span>|\n)/)) {
    if (!token) continue;
    if (token === "\n") {
      lines.push(current + "</span>".repeat(open.length));
      current = open.join("");
    } else if (token.startsWith("<span")) {
      open.push(token);
      current += token;
    } else if (token === "</span>") {
      open.pop();
      current += token;
    } else {
      current += token;
    }
  }
  lines.push(current);
  return lines
    .map((line, index) => {
      const text = sourceLines[index] ?? "";
      const indent = text.length - text.trimStart().length + 4;
      // A blank line, even one inside a docstring's markup, still takes a line.
      const body = text.trim() ? line : `${line}&#8203;`;
      return `<span class="ln" style="--indent:${indent}ch">${body}</span>`;
    })
    .join("");
}

/** A whole Python file, highlighted with the site's code colours. */
function CodeFile({ name, code }: { name: string; code: string }): ReactNode {
  const source = code.replace(/\s+$/, "");
  const html = splitLines(hljs.highlight(source, { language: "python", ignoreIllegals: true }).value, source);
  return (
    <div className="lesson codefile">
      <div className="codeblock">
        <div className="codeblock-bar">
          <span>{name}</span>
          <span className="codefile-lines">{source.split("\n").length} lines</span>
        </div>
        <pre>
          <code className="hljs language-python" dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
      </div>
    </div>
  );
}
