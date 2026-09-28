import { useEffect, useState } from "react";

/**
 * Tracks a CSS media query from React.
 *
 * The layout mostly responds through Tailwind's own breakpoints, but the
 * lesson page has to *choose a component* rather than a class: a draggable
 * split makes no sense on a phone. This reads the same breakpoint from
 * JavaScript so the two never disagree.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    const list = window.matchMedia(query);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, [query]);

  return matches;
}

/** Tailwind's `lg`, the width at which the lesson and the editor sit side by side. */
export const WIDE = "(min-width: 1024px)";
