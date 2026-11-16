import Editor, { type OnMount } from "@monaco-editor/react";
import { useEffect, useRef, useState } from "react";

import {
  definePaperTheme,
  editorLook,
  editorThemeName,
  fontHasLigatures,
  fontStack,
  useEditorPrefs,
  useMonacoPrefs,
} from "../lib/editorPrefs";
import { REVEAL_EVENT, type RevealDetail } from "../lib/events";
import { EDITOR_OPTIONS, THEMES, monaco as bundledMonaco } from "../lib/monaco";
import { useTheme } from "../lib/theme";
import { WIDE, useMediaQuery } from "../lib/useMediaQuery";

type Props = {
  value: string;
  onChange: (next: string) => void;
  onRun: () => void;
  onCheck: () => void;
  readOnly?: boolean;
};

definePaperTheme(bundledMonaco);

type MountedEditor = Parameters<OnMount>[0];
type MountedMonaco = Parameters<OnMount>[1];

/**
 * The Python editor. Cmd or Ctrl with Return runs the file; adding Shift runs
 * the checks, which matches the toolbar's Run and Check buttons. The theme,
 * font, size, and Vim mode come from the shared editor prefs.
 */
export default function CodeEditor({ value, onChange, onRun, onCheck, readOnly = false }: Props) {
  // A phone fits about 35 characters across, so unwrapped code spends most of
  // its time off the right edge. Wide screens keep the lines as written.
  const wide = useMediaQuery(WIDE);
  const site = useTheme().resolved;
  const [prefs] = useEditorPrefs();
  const theme = editorThemeName(editorLook(prefs.theme, site), THEMES);

  const [editor, setEditor] = useState<MountedEditor | null>(null);
  const [monaco, setMonaco] = useState<MountedMonaco | null>(null);
  const [vimStatus, setVimStatus] = useState<HTMLDivElement | null>(null);
  const flash = useRef<{ clear: () => void } | null>(null);

  useMonacoPrefs({ editor, monaco, site, themes: THEMES, vimStatus, onRun, onCheck });

  // A click on a function name in the lesson, or on "Start coding", lands
  // here: find the line, scroll it to the middle, and light it up briefly.
  useEffect(() => {
    function onReveal(event: Event) {
      const { symbol } = (event as CustomEvent<RevealDetail>).detail ?? {};
      // On a phone the editor was hidden until this event switched tabs, so
      // wait a frame for it to have a size.
      window.setTimeout(() => {
        const model = editor?.getModel();
        if (!editor || !model) return;
        editor.layout();
        const lines = model.getLinesContent();
        const pattern = symbol ? new RegExp(`^\\s*(?:def|class)\\s+${symbol}\\b`) : /raise NotImplementedError/;
        let index = lines.findIndex((line) => pattern.test(line));
        if (index < 0 && !symbol) index = 0;
        if (index < 0) return;
        const line = index + 1;
        editor.revealLineInCenter(line);
        editor.setPosition({ lineNumber: line, column: lines[index].search(/\S|$/) + 1 });
        editor.focus();
        flash.current?.clear();
        const decorations = editor.createDecorationsCollection([
          {
            range: { startLineNumber: line, startColumn: 1, endLineNumber: line, endColumn: 1 },
            options: { isWholeLine: true, className: "reveal-line", linesDecorationsClassName: "reveal-gutter" },
          },
        ]);
        flash.current = decorations;
        window.setTimeout(() => decorations.clear(), 1800);
      }, 60);
    }
    window.addEventListener(REVEAL_EVENT, onReveal);
    return () => window.removeEventListener(REVEAL_EVENT, onReveal);
  }, [editor]);

  const mount: OnMount = (mounted, instance) => {
    setEditor(mounted);
    setMonaco(instance);
    mounted.focus();
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1">
        <Editor
          language="python"
          theme={theme}
          value={value}
          onChange={(next) => onChange(next ?? "")}
          onMount={mount}
          options={{
            ...EDITOR_OPTIONS,
            readOnly,
            fontSize: prefs.fontSize,
            lineHeight: Math.round(prefs.fontSize * 1.55),
            fontFamily: fontStack(prefs.fontFamily),
            fontLigatures: fontHasLigatures(prefs.fontFamily),
            wordWrap: wide ? "off" : "on",
            // The 88-column ruler is meaningless once the lines wrap.
            rulers: wide ? EDITOR_OPTIONS.rulers : [],
            lineNumbersMinChars: wide ? 5 : 3,
            folding: wide,
          }}
          loading={<div className="flex h-full items-center justify-center text-[13px] text-ink-4">Loading the editor…</div>}
        />
      </div>
      {prefs.vim && <div ref={setVimStatus} className="sh-vim-status" aria-live="polite" aria-label="Vim status" />}
    </div>
  );
}
