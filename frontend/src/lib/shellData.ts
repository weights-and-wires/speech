/**
 * Builds the shared shell components' props from the speech course's lesson
 * index. This file is course-specific: each course writes its own version.
 */

import type { SearchEntry } from "../components/SearchDialog";
import type { LessonMeta, Part, ProgressMap } from "./api";
import { sectionId } from "./lessonOutline";
import { lessonNumbers, splitPart } from "./parts";

/** Every lesson as a search entry, with `##` and `###` headings. */
export function searchEntries(parts: Part[]): SearchEntry[] {
  const all = parts.flatMap((part) => part.lessons);
  const numbers = lessonNumbers(all);
  return parts.flatMap((part) =>
    part.lessons.map((lesson) => {
      const href = `/lessons/${lesson.slug}`;
      // The reader anchors `##` sections only, so a `###` heading links to
      // the `##` section above it.
      let parent = "";
      const headings = lesson.headings.map((heading) => {
        if (heading.level === 2) parent = sectionId(heading.text);
        return { text: heading.text, href: parent ? `${href}#${parent}` : href };
      });
      return {
        id: lesson.slug,
        href,
        title: lesson.title,
        kicker: `Lesson ${numbers.get(lesson.slug) ?? ""} · ${splitPart(part.part).name}`,
        summary: lesson.summary,
        keywords: [...lesson.objectives, part.part, lesson.slug],
        headings,
      };
    }),
  );
}

/**
 * The lesson to open next: the one touched most recently that is not yet
 * solved, then the first one never opened, then the first lesson.
 */
export function nextLesson(lessons: LessonMeta[], progress: ProgressMap): LessonMeta | undefined {
  const started = lessons
    .filter((l) => progress[l.slug]?.status === "started")
    .sort((a, b) => (progress[b.slug]?.updated_at ?? 0) - (progress[a.slug]?.updated_at ?? 0));
  return started[0] ?? lessons.find((l) => !progress[l.slug]) ?? lessons[0];
}

export function solvedCount(lessons: LessonMeta[], progress: ProgressMap): number {
  return lessons.filter((l) => progress[l.slug]?.status === "solved").length;
}

/** "Part 3 — Classical recognition" → "03", or the 1-based index when unnumbered. */
export function partNumber(part: string, index: number): string {
  return (splitPart(part).number || String(index + 1)).padStart(2, "0");
}
