import Editor, { type OnMount } from "@monaco-editor/react";
import { useEffect, useRef } from "react";

import { REVEAL_EVENT, type RevealDetail } from "../lib/events";
import { EDITOR_OPTIONS, THEMES } from "../lib/monaco";
import { useTheme } from "../lib/theme";
import { WIDE, useMediaQuery } from "../lib/useMediaQuery";

type Props = {
  value: string;
  onChange: (next: string) => void;
  onRun: () => void;
  onCheck: () => void;
  readOnly?: boolean;
};

/**
 * The Python editor. Cmd or Ctrl with Return runs the file; adding Shift runs
 * the checks, which matches what the two buttons above it do.
 */
export default function CodeEditor({
  value,
  onChange,
  onRun,
  onCheck,
  readOnly = false,
}: Props) {
  // A phone fits about 35 characters across, so unwrapped code spends most of
  // its time off the right edge. Wide screens keep the lines as written.
  const wide = useMediaQuery(WIDE);
  const theme = THEMES[useTheme().resolved];
  const themeRef = useRef(theme);
  themeRef.current = theme;

  // The keyboard commands are registered once, against a ref, so that changing
  // the handler between renders does not re-register them.
  const run = useRef(onRun);
  const check = useRef(onCheck);
  run.current = onRun;
  check.current = onCheck;

  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);
  const flash = useRef<{ clear: () => void } | null>(null);

  // A click on a function name in the lesson, or on "Start coding", lands
  // here: find the line, scroll it to the middle, and light it up briefly.
  useEffect(() => {
    function onReveal(event: Event) {
      const { symbol } = (event as CustomEvent<RevealDetail>).detail ?? {};
      // On a phone the editor was hidden until this event switched tabs, so
      // wait a frame for it to have a size.
      window.setTimeout(() => {
        const editor = editorRef.current;
        const model = editor?.getModel();
        if (!editor || !model) return;
        editor.layout();
        const lines = model.getLinesContent();
        const pattern = symbol
          ? new RegExp(`^\\s*(?:def|class)\\s+${symbol}\\b`)
          : /raise NotImplementedError/;
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
  }, []);

  const mount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monaco.editor.setTheme(themeRef.current);
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () =>
      run.current(),
    );
    editor.addCommand(
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.Enter,
      () => check.current(),
    );
    editor.focus();
  };

  return (
    <Editor
      language="python"
      theme={theme}
      value={value}
      onChange={(next) => onChange(next ?? "")}
      onMount={mount}
      options={{
        ...EDITOR_OPTIONS,
        readOnly,
        wordWrap: wide ? "off" : "on",
        // The 88-column ruler is meaningless once the lines wrap.
        rulers: wide ? EDITOR_OPTIONS.rulers : [],
        lineNumbersMinChars: wide ? 5 : 3,
        folding: wide,
      }}
      loading={
        <div className="flex h-full items-center justify-center text-[13px] text-ink-4">
          Loading the editor…
        </div>
      }
    />
  );
}
