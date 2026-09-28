import { useEffect, useState } from "react";

/**
 * Light, dark, or whatever the system prefers. The choice is stored, applied
 * to <html data-theme>, and broadcast so the editor can follow it. index.html
 * applies the stored choice before the first paint, so a dark reader never
 * sees a white flash.
 */

export type ThemeMode = "system" | "light" | "dark";
export type Resolved = "light" | "dark";

const KEY = "speech.theme";
const EVENT = "speech:theme";
const DARK = "(prefers-color-scheme: dark)";

function stored(): ThemeMode {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

function resolve(mode: ThemeMode): Resolved {
  if (mode !== "system") return mode;
  return window.matchMedia(DARK).matches ? "dark" : "light";
}

function apply(resolved: Resolved) {
  const root = document.documentElement;
  root.dataset.theme = resolved;
  root.style.colorScheme = resolved;
}

export function useTheme() {
  const [mode, setModeState] = useState<ThemeMode>(stored);
  const [resolved, setResolved] = useState<Resolved>(() => resolve(stored()));

  useEffect(() => {
    const list = window.matchMedia(DARK);
    const update = () => {
      const next = resolve(stored());
      apply(next);
      setResolved(next);
      setModeState(stored());
    };
    update();
    list.addEventListener("change", update);
    window.addEventListener(EVENT, update);
    return () => {
      list.removeEventListener("change", update);
      window.removeEventListener(EVENT, update);
    };
  }, []);

  function setMode(next: ThemeMode) {
    try {
      if (next === "system") window.localStorage.removeItem(KEY);
      else window.localStorage.setItem(KEY, next);
    } catch {
      // Without storage the choice lasts until the page reloads.
    }
    const value = resolve(next);
    apply(value);
    setModeState(next);
    setResolved(value);
    window.dispatchEvent(new Event(EVENT));
  }

  return { mode, resolved, setMode };
}
