/**
 * Gives a rendered lesson its shape. Runs after KaTeX and highlight.js, on the
 * flat list of blocks Markdown produces, and:
 *
 * - wraps each `##` section in a <section> that knows its kind and number;
 * - turns `> [!KEY]`, `[!TIP]`, `[!WARNING]`, and `[!RECAP]` blockquotes into
 *   labelled callouts;
 * - frames display maths, and the "Read it aloud" list that explains it;
 * - boxes each worked example;
 * - marks defined terms, run-in headings, and "common mistake" phrases; and
 * - tags inline code that names a function in the exercise, so the reader can
 *   jump to it in the editor.
 */

import { KIND_LABEL, isTerm, sectionId, sectionKind, termId } from "./lessonOutline";

type Text = { type: "text"; value: string };
type Element = {
  type: "element";
  tagName: string;
  properties: Record<string, unknown>;
  children: Node[];
};
type Node = Text | Element | { type: string; children?: Node[]; value?: string };
type Root = { type: "root"; children: Node[] };

const CALLOUT_LABEL: Record<string, string> = {
  KEY: "Key idea",
  TIP: "Tip",
  WARNING: "Watch out",
  RECAP: "Before you build",
};

function el(tagName: string, properties: Record<string, unknown>, children: Node[]): Element {
  return { type: "element", tagName, properties, children };
}

function isElement(node: Node | undefined, tag?: string): node is Element {
  return !!node && node.type === "element" && (!tag || (node as Element).tagName === tag);
}

function classes(node: Element): string[] {
  const value = node.properties.className;
  return Array.isArray(value) ? (value as string[]) : typeof value === "string" ? [value] : [];
}

function addClass(node: Element, name: string) {
  node.properties.className = [...classes(node), name];
}

function textOf(node: Node): string {
  if (node.type === "text") return (node as Text).value;
  return ((node as Element).children ?? []).map(textOf).join("");
}

function isBlank(node: Node): boolean {
  return node.type === "text" && !(node as Text).value.trim();
}

/** Blockquote → callout, when its first text starts with a `[!TYPE]` marker. */
function callout(node: Element): Element | null {
  const first = node.children.find((c) => !isBlank(c));
  if (!isElement(first, "p")) return null;
  const lead = first.children[0];
  if (!lead || lead.type !== "text") return null;
  const match = /^\s*\[!(\w+)\]\s*/.exec((lead as Text).value);
  if (!match) return null;
  const kind = match[1].toUpperCase();
  const label = CALLOUT_LABEL[kind];
  if (!label) return null;

  (lead as Text).value = (lead as Text).value.slice(match[0].length);
  const body = node.children.filter(
    (c) => !(c === first && !textOf(first).trim() && first.children.length <= 1),
  );
  return el("aside", { className: ["callout", `callout-${kind.toLowerCase()}`], role: "note" }, [
    el("div", { className: ["callout-label"] }, [{ type: "text", value: label }]),
    el("div", { className: ["callout-body"] }, body),
  ]);
}

const MISTAKE = /(The (?:most )?common mistakes? (?:here )?(?:is|are)|Common mistakes?:|A common mistake is)\s*/i;

/** Walks inline content: terms, run-in headings, mistakes, and code symbols. */
function decorate(
  node: Element,
  ctx: { symbols: Set<string>; seenTerms: Set<string> },
) {
  const next: Node[] = [];
  node.children.forEach((child, index) => {
    if (child.type === "text") {
      const value = (child as Text).value;
      const match = MISTAKE.exec(value);
      if (match && node.tagName !== "code") {
        const before = value.slice(0, match.index);
        const after = value.slice(match.index + match[0].length);
        if (before) next.push({ type: "text", value: before });
        next.push(el("span", { className: ["mistake-tag"] }, [{ type: "text", value: "Common mistake" }]));
        if (after) next.push({ type: "text", value: after });
        return;
      }
      next.push(child);
      return;
    }
    if (!isElement(child)) {
      next.push(child);
      return;
    }
    if (child.tagName === "strong") {
      const text = textOf(child).trim();
      const firstContent = node.children.findIndex((c) => !isBlank(c)) === index;
      if (firstContent && /[.:]$/.test(text) && (node.tagName === "p" || node.tagName === "li")) {
        addClass(child, "runin");
      } else if (isTerm(text) && !child.children.some((c) => isElement(c, "code"))) {
        // Bold code names a function, not a term, which is also how the
        // flashcards in lessonOutline tell the two apart.
        addClass(child, "term");
        const key = text.toLowerCase();
        if (!ctx.seenTerms.has(key)) {
          ctx.seenTerms.add(key);
          child.properties.id = termId(text);
        }
      }
    }
    if (child.tagName === "code" && node.tagName !== "pre") {
      const name = textOf(child).split("(")[0].split(".").pop()?.trim() ?? "";
      if (ctx.symbols.has(name)) child.properties.dataSymbol = name;
    }
    if (!classes(child).some((c) => c.startsWith("katex"))) decorate(child, ctx);
    next.push(child);
  });
  node.children = next;
}

/** Boxes each `### Worked example` with the blocks that follow it. */
function boxWorkedExamples(children: Node[]): Node[] {
  const out: Node[] = [];
  let box: Element | null = null;
  for (const child of children) {
    if (isElement(child) && /^h[1-3]$/.test(child.tagName)) {
      box = null;
      if (child.tagName === "h3" && /worked example|by hand/i.test(textOf(child))) {
        box = el("div", { className: ["worked"] }, [
          el("div", { className: ["worked-label"] }, [{ type: "text", value: "Try it by hand" }]),
          child,
        ]);
        out.push(box);
        continue;
      }
    }
    // A callout sums up the section, so it closes the example before it.
    if (isElement(child, "aside") && classes(child).includes("callout")) box = null;
    if (box) box.children.push(child);
    else out.push(child);
  }
  return out;
}

/** Frames display maths, and ties a following "Read it aloud" list to it. */
function frameFormulas(children: Node[]): Node[] {
  const out: Node[] = [];
  let lastFormula: Element | null = null;
  let expectLegend = false;
  for (const child of children) {
    if (isBlank(child)) {
      out.push(child);
      continue;
    }
    if (isElement(child, "span") && classes(child).includes("katex-display")) {
      lastFormula = el("div", { className: ["formula"] }, [child]);
      out.push(lastFormula);
      expectLegend = false;
      continue;
    }
    if (isElement(child, "p") && /^read (it|this|each|the formula)[^.]*aloud/i.test(textOf(child).trim())) {
      addClass(child, "readaloud-lead");
      if (lastFormula) {
        lastFormula.children.push(child);
        expectLegend = true;
        continue;
      }
      out.push(child);
      expectLegend = true;
      continue;
    }
    if (expectLegend && isElement(child) && (child.tagName === "ul" || child.tagName === "ol")) {
      addClass(child, "readaloud");
      if (lastFormula && lastFormula.children.some((c) => isElement(c) && classes(c).includes("readaloud-lead"))) {
        lastFormula.children.push(child);
      } else {
        out.push(child);
      }
      expectLegend = false;
      lastFormula = null;
      continue;
    }
    expectLegend = false;
    lastFormula = null;
    out.push(child);
  }
  return out;
}

export default function rehypeLesson(options: { symbols?: string[] } = {}) {
  const ctx = { symbols: new Set(options.symbols ?? []), seenTerms: new Set<string>() };
  return (tree: Root) => {
    ctx.seenTerms.clear();

    // Callouts first, so what they hold is decorated like any other text.
    tree.children = tree.children.map((child) =>
      isElement(child, "blockquote") ? callout(child) ?? child : child,
    );
    for (const child of tree.children) if (isElement(child)) decorate(child, ctx);

    const out: Node[] = [];
    let intro: Element | null = null;
    let current: Element | null = null;
    let number = 0;
    for (const child of tree.children) {
      // The reader shows the title itself, above the objectives.
      if (isElement(child, "h1")) {
        intro = el("div", { className: ["lesson-intro"] }, []);
        out.push(intro);
        continue;
      }
      if (isElement(child, "h2")) {
        const title = textOf(child).trim();
        const kind = sectionKind(title);
        number += 1;
        child.children = [
          el("span", { className: ["h2-kicker"], ariaHidden: "true" }, [
            el("span", { className: ["h2-num"] }, [
              { type: "text", value: String(number) },
            ]),
            el("span", { className: ["h2-kind"] }, [{ type: "text", value: KIND_LABEL[kind] }]),
          ]),
          el("span", { className: ["h2-text"] }, child.children),
        ];
        current = el(
          "section",
          { className: ["lesson-section"], id: sectionId(title), dataKind: kind, dataSection: "" },
          [child],
        );
        out.push(current);
        continue;
      }
      if (current) current.children.push(child);
      else if (intro) intro.children.push(child);
      else out.push(child);
    }

    for (const node of out) {
      if (isElement(node, "section") || (isElement(node) && classes(node).includes("lesson-intro"))) {
        node.children = frameFormulas(boxWorkedExamples(node.children));
        for (const child of node.children) {
          if (isElement(child, "div") && classes(child).includes("worked")) {
            child.children = frameFormulas(child.children);
          }
        }
      }
    }

    // Each run-in heading in "Where this breaks" becomes its own card,
    // whether the lesson writes them as paragraphs or as a list.
    const opensWithRunin = (node: Element) => {
      const first = node.children.find((c) => !isBlank(c));
      return isElement(first, "strong") && classes(first).includes("runin");
    };
    for (const node of out) {
      if (!isElement(node, "section") || node.properties.dataKind !== "pitfalls") continue;
      for (const child of node.children) {
        if (isElement(child, "p") && opensWithRunin(child)) addClass(child, "pitfall");
        if (isElement(child, "ul") || isElement(child, "ol")) {
          const items = child.children.filter((c): c is Element => isElement(c, "li"));
          if (items.length && items.every(opensWithRunin)) {
            addClass(child, "pitfall-list");
            items.forEach((item) => addClass(item, "pitfall"));
          }
        }
      }
    }

    tree.children = out;
  };
}
