import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";

import DownloadBook from "./components/DownloadBook";
import Login from "./components/Login";
import Mark from "./components/Mark";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import DemoGallery from "./pages/DemoGallery";
import LessonPage from "./pages/LessonPage";
import { api, type Part, type ProgressMap } from "./lib/api";
import { useTheme, type ThemeMode } from "./lib/theme";

const COLLAPSED_KEY = "speech.sidebar.collapsed";

function rememberedCollapse(): boolean {
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export default function App() {
  const [username, setUsername] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [parts, setParts] = useState<Part[]>([]);
  const [progress, setProgress] = useState<ProgressMap>({});
  const [gpu, setGpu] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  // menuOpen drives the drawer on a narrow screen; collapsed hides the lesson
  // list on a wide one, which a reader wants once the editor is open.
  const [collapsed, setCollapsed] = useState(rememberedCollapse);
  const location = useLocation();
  const lessons = useMemo(() => parts.flatMap((part) => part.lessons), [parts]);

  useEffect(() => {
    api
      .session()
      .then((body) => setUsername(body.username))
      .catch(() => setUsername(null))
      .finally(() => setReady(true));
  }, []);

  const refresh = useCallback(() => {
    if (!username) return;
    api
      .lessons()
      .then((body) => {
        setParts(body.parts);
        setProgress(body.progress);
      })
      .catch(() => undefined);
  }, [username]);

  useEffect(() => {
    refresh();
    if (username) api.health().then((body) => setGpu(body.gpu)).catch(() => undefined);
  }, [username, refresh]);

  useEffect(() => setMenuOpen(false), [location.pathname]);

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // Storage being off only costs the reader their sidebar state.
    }
  }, [collapsed]);

  // Cmd or Ctrl with B shows and hides the lesson list, the way an editor
  // treats its own sidebar. Monaco leaves this combination alone.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      if (event.key !== "b" && event.key !== "B") return;
      event.preventDefault();
      setCollapsed((v) => !v);
      setMenuOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!ready) {
    return <div className="h-full bg-canvas" aria-busy="true" />;
  }

  if (!username) {
    return <Login />;
  }

  const onLesson = location.pathname.startsWith("/lessons/");

  return (
    <div className="flex h-full flex-col bg-canvas">
      <header className="z-40 flex h-12 shrink-0 items-center gap-1 border-b border-rule bg-chrome px-2 lg:h-11">
        <button
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label={menuOpen ? "Hide the lesson list" : "Show the lesson list"}
          aria-expanded={menuOpen}
          className="icon-btn lg:hidden"
        >
          <MenuIcon />
        </button>

        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          aria-label={collapsed ? "Show the lesson list" : "Hide the lesson list"}
          aria-expanded={!collapsed}
          title={`${collapsed ? "Show" : "Hide"} the lesson list (Cmd or Ctrl with B)`}
          className="icon-btn hidden lg:inline-flex"
        >
          <PanelIcon open={!collapsed} />
        </button>

        <Link
          to="/"
          className="ml-1 flex min-h-10 items-center gap-2 rounded-md px-1.5 transition-opacity hover:opacity-80 lg:min-h-8"
          aria-label="Speech and audio, course contents"
        >
          <Mark size={20} />
          <span className="font-serif text-[15.5px] font-semibold tracking-tight text-ink">
            Speech and audio
          </span>
        </Link>

        {onLesson && (
          <Link
            to="/"
            className="ml-1 hidden rounded-md px-2 py-1 text-[13px] text-ink-3 transition-colors hover:text-ink sm:block"
          >
            Contents
          </Link>
        )}

        <div className="flex-1" />

        {gpu && (
          <span
            title="A RunPod GPU endpoint is configured"
            className="mr-2 hidden items-center gap-1.5 text-[12px] text-ink-3 sm:inline-flex"
          >
            <span className="size-1.5 rounded-full bg-good" aria-hidden />
            GPU available
          </span>
        )}

        <DownloadBook />

        <AccountMenu
          username={username}
          onSignOut={() => {
            const portal = new URL("https://auth.ifkash.dev/logout");
            portal.searchParams.set("rd", window.location.href);
            window.location.assign(portal.href);
          }}
        />
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar
          parts={parts}
          progress={progress}
          open={menuOpen}
          collapsed={collapsed}
          onClose={() => setMenuOpen(false)}
          onCollapse={() => setCollapsed(true)}
        />

        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <Routes>
            <Route
              path="/"
              element={<Dashboard parts={parts} progress={progress} />}
            />
            <Route
              path="/lessons/:slug"
              element={
                <LessonPage gpuAvailable={gpu} lessons={lessons} onProgress={refresh} />
              }
            />
            <Route path="/demos" element={<DemoGallery />} />
            <Route path="/demos/:id" element={<DemoGallery />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

const THEMES: { mode: ThemeMode; label: string }[] = [
  { mode: "system", label: "Match the system" },
  { mode: "light", label: "Light" },
  { mode: "dark", label: "Dark" },
];

/** The reader's name, with the theme choice and sign-out behind it. */
function AccountMenu({ username, onSignOut }: { username: string; onSignOut: () => void }) {
  const [open, setOpen] = useState(false);
  const { mode, setMode } = useTheme();
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointer(event: PointerEvent) {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    window.addEventListener("pointerdown", onPointer);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
        className="flex min-h-10 items-center gap-2 rounded-md px-1.5 text-[13px] text-ink-2 transition-colors hover:bg-[color-mix(in_oklab,var(--ink)_6%,transparent)] lg:min-h-8"
      >
        <span className="flex size-6 items-center justify-center rounded-full bg-primary text-[11px] font-semibold uppercase text-on-primary">
          {username.slice(0, 1)}
        </span>
        <span className="hidden sm:inline">{username}</span>
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden className="text-ink-4">
          <path d="M2.5 3.75 5 6.25l2.5-2.5" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div role="menu" className="menu fade-in absolute right-0 top-[calc(100%+6px)] z-50 w-56">
          <p className="px-2.5 pb-1 pt-1.5 text-[11.5px] text-ink-4">Signed in as {username}</p>
          <div className="my-1 border-t border-rule" />
          <p className="px-2.5 pb-1 pt-1 text-[11.5px] font-medium text-ink-3">Appearance</p>
          {THEMES.map((theme) => (
            <button
              key={theme.mode}
              type="button"
              role="menuitemradio"
              aria-checked={mode === theme.mode}
              onClick={() => setMode(theme.mode)}
              className="menu-item"
            >
              <span className="w-3 text-accent" aria-hidden>
                {mode === theme.mode ? "✓" : ""}
              </span>
              {theme.label}
            </button>
          ))}
          <div className="my-1 border-t border-rule" />
          <button type="button" role="menuitem" onClick={onSignOut} className="menu-item">
            <span className="w-3" aria-hidden />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}

function MenuIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
      <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

/** A pane outline whose left column fills in when the lesson list is showing. */
function PanelIcon({ open }: { open: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <rect x="1.75" y="2.75" width="12.5" height="10.5" rx="2" stroke="currentColor" strokeWidth="1.3" />
      <path d="M6.25 3v10" stroke="currentColor" strokeWidth="1.3" />
      {open && <rect x="2.5" y="3.5" width="3.1" height="9" rx="0.8" fill="currentColor" opacity="0.35" />}
    </svg>
  );
}
