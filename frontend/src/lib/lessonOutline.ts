/**
 * Reads the structure a lesson's Markdown already has: its `##` sections and
 * the bold terms it defines. The reader uses the sections for the progress
 * bar and the contents menu, and the terms for the flashcards at the end.
 */

export type SectionKind =
  | "recap"
  | "concept"
  | "example"
  | "build"
  | "cost"
  | "pitfalls"
  | "exercise";

export type Section = { id: string; title: string; kind: SectionKind };

export type Term = { id: string; term: string; sentence: string };

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_$\\]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function sectionId(title: string): string {
  return `s-${slugify(title)}`;
}

export function termId(term: string): string {
  return `term-${slugify(term)}`;
}

export function sectionKind(title: string): SectionKind {
  const t = title.trim().toLowerCase();
  if (t.startsWith("what you already know")) return "recap";
  if (t.startsWith("build it step by step")) return "build";
  if (t.startsWith("cost and scaling")) return "cost";
  if (t.startsWith("where this breaks")) return "pitfalls";
  if (t.startsWith("the exercise")) return "exercise";
  if (/worked example|by hand/.test(t)) return "example";
  return "concept";
}

/** The label shown above a section heading, so each kind reads differently. */
export const KIND_LABEL: Record<SectionKind, string> = {
  recap: "Warm-up",
  concept: "Concept",
  example: "Try it by hand",
  build: "Hands-on",
  cost: "Under the hood",
  pitfalls: "Pitfalls",
  exercise: "Your turn",
};

function withoutCode(markdown: string): string {
  return markdown.replace(/```[\s\S]*?```/g, "");
}

export function sections(markdown: string): Section[] {
  const out: Section[] = [];
  for (const line of withoutCode(markdown).split("\n")) {
    const match = /^## (.+?)\s*$/.exec(line);
    if (!match) continue;
    out.push({ id: sectionId(match[1]), title: match[1], kind: sectionKind(match[1]) });
  }
  return out;
}

/**
 * A bold run of one to four words, with no code and no closing punctuation, is
 * a term the lesson defines. A bold phrase that ends in a full stop is a
 * run-in heading instead, and bold code is a function name.
 */
export function isTerm(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed || /[`.:?!]$/.test(trimmed) || trimmed.includes("`")) return false;
  const words = trimmed.split(/\s+/).length;
  return words <= 4 && trimmed.length <= 40;
}

/** The terms a lesson defines, each with the sentence that defines it. */
export function terms(markdown: string): Term[] {
  const seen = new Set<string>();
  const out: Term[] = [];
  const blocks = withoutCode(markdown).split(/\n\s*\n/);
  for (const raw of blocks) {
    const block = raw.trim();
    if (!block || /^(#|>|\||\$\$)/.test(block)) continue;
    // One list item per line, so each keeps its own sentence.
    const lines = block.split("\n").some((l) => /^\s*([-*]|\d+\.)\s/.test(l))
      ? block.split(/\n(?=\s*(?:[-*]|\d+\.)\s)/)
      : [block.replace(/\n/g, " ")];
    for (const line of lines) {
      const listItem = /^\s*([-*]|\d+\.)\s/.test(line);
      const text = line.replace(/\n/g, " ").replace(/^\s*([-*]|\d+\.)\s+/, "");
      // A list item is usually one short definition, so it stays whole. A
      // paragraph is split, so the card shows only the sentence that matters.
      const sentences = listItem ? [text] : text.split(/(?<=[.!?])\s+(?=[A-Z*$`])/);
      for (const sentence of sentences) {
        for (const match of sentence.matchAll(/\*\*([^*]+)\*\*/g)) {
          const term = match[1].trim();
          const key = term.toLowerCase();
          if (!isTerm(term) || seen.has(key)) continue;
          // A term with no sentence around it is a label, not a definition.
          if (sentence.replace(match[0], "").trim().split(/\s+/).length < 4) continue;
          seen.add(key);
          out.push({ id: termId(term), term, sentence: sentence.trim() });
        }
      }
    }
  }
  return out;
}

/** The function and class names a starter file defines. */
export function symbols(starter: string): string[] {
  const names = new Set<string>();
  for (const match of starter.matchAll(/^\s*(?:def|class)\s+([A-Za-z_]\w*)/gm)) {
    names.add(match[1]);
  }
  return [...names];
}
