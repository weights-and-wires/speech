/**
 * Monaco, bundled with the app instead of fetched from a CDN at run time.
 *
 * `@monaco-editor/react` loads the editor from jsdelivr unless it is handed an
 * instance. The app is behind a login on a private host, so it builds a lean
 * instance with the features and the one language the exercises need, gives it
 * the app's colours in a light and a dark theme, and registers it with the
 * loader.
 */

import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/editor/editor.api";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";

// Editor features. The full set adds a diff editor, a colour picker, code
// lenses, and other machinery that has no provider here.
import "monaco-editor/features/codicon/register";
import "monaco-editor/features/bracketMatching/register";
import "monaco-editor/features/caretOperations/register";
import "monaco-editor/features/clipboard/register";
import "monaco-editor/features/comment/register";
import "monaco-editor/features/contextmenu/register";
import "monaco-editor/features/cursorUndo/register";
import "monaco-editor/features/find/register";
import "monaco-editor/features/folding/register";
import "monaco-editor/features/indentation/register";
import "monaco-editor/features/lineSelection/register";
import "monaco-editor/features/linesOperations/register";
import "monaco-editor/features/longLinesHelper/register";
import "monaco-editor/features/multicursor/register";
import "monaco-editor/features/smartSelect/register";
import "monaco-editor/features/suggest/register";
import "monaco-editor/features/tokenization/register";
import "monaco-editor/features/wordHighlighter/register";
import "monaco-editor/features/wordOperations/register";
import "monaco-editor/features/wordPartOperations/register";

// The one language the exercises are written in.
import "monaco-editor/languages/definitions/python/register";

self.MonacoEnvironment = {
  // Python has no language service worker of its own, so the base editor
  // worker is the only one this app needs.
  getWorker: () => new EditorWorker(),
};

export const THEMES = { light: "ww-light", dark: "ww-dark" } as const;

// Both themes mirror the tokens in index.css, so lesson code blocks and the
// editor colour the same token the same way.
monaco.editor.defineTheme(THEMES.light, {
  base: "vs",
  inherit: true,
  rules: [
    { token: "", foreground: "1c1b18" },
    { token: "keyword", foreground: "7b3fa0" },
    { token: "string", foreground: "2d7a36" },
    { token: "string.escape", foreground: "1f5e27" },
    { token: "comment", foreground: "8c887d", fontStyle: "italic" },
    { token: "number", foreground: "a85300" },
    { token: "type", foreground: "0d6b78" },
    { token: "delimiter", foreground: "6b675d" },
    { token: "identifier", foreground: "1c1b18" },
  ],
  colors: {
    "editor.background": "#fcfbf8",
    "editor.foreground": "#1c1b18",
    "editorCursor.foreground": "#2b59c3",
    "editor.lineHighlightBackground": "#f5f3ee",
    "editor.lineHighlightBorder": "#f5f3ee00",
    "editor.selectionBackground": "#2b59c333",
    "editor.inactiveSelectionBackground": "#2b59c31a",
    "editor.selectionHighlightBackground": "#2b59c31a",
    "editor.wordHighlightBackground": "#2b59c31a",
    "editor.findMatchBackground": "#f2c94c80",
    "editor.findMatchHighlightBackground": "#f2c94c40",
    "editorLineNumber.foreground": "#c2beb2",
    "editorLineNumber.activeForeground": "#6b675d",
    "editorIndentGuide.background1": "#ebe8e0",
    "editorIndentGuide.activeBackground1": "#d4d0c4",
    "editorWhitespace.foreground": "#e6e3da",
    "editorBracketMatch.background": "#2b59c31a",
    "editorBracketMatch.border": "#2b59c366",
    "editorBracketHighlight.foreground1": "#6b675d",
    "editorBracketHighlight.foreground2": "#7b3fa0",
    "editorBracketHighlight.foreground3": "#0d6b78",
    "editorGutter.background": "#fcfbf8",
    "editorWidget.background": "#ffffff",
    "editorWidget.border": "#e6e3da",
    "editorSuggestWidget.background": "#ffffff",
    "editorSuggestWidget.border": "#e6e3da",
    "editorSuggestWidget.selectedBackground": "#f0eee7",
    "editorSuggestWidget.highlightForeground": "#2b59c3",
    "editorHoverWidget.background": "#ffffff",
    "editorHoverWidget.border": "#e6e3da",
    "editorRuler.foreground": "#efece5",
    focusBorder: "#2b59c3",
    "scrollbar.shadow": "#00000000",
    "scrollbarSlider.background": "#d4d0c488",
    "scrollbarSlider.hoverBackground": "#bdb8aa99",
    "scrollbarSlider.activeBackground": "#bdb8aacc",
    "editorOverviewRuler.border": "#00000000",
  },
});

monaco.editor.defineTheme(THEMES.dark, {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "", foreground: "eeece6" },
    { token: "keyword", foreground: "c9a3ee" },
    { token: "string", foreground: "a3d69c" },
    { token: "string.escape", foreground: "c6e8c1" },
    { token: "comment", foreground: "7f7c72", fontStyle: "italic" },
    { token: "number", foreground: "efb46c" },
    { token: "type", foreground: "7fd0d8" },
    { token: "delimiter", foreground: "a19d92" },
    { token: "identifier", foreground: "eeece6" },
  ],
  colors: {
    "editor.background": "#1a1a18",
    "editor.foreground": "#eeece6",
    "editorCursor.foreground": "#8cadff",
    "editor.lineHighlightBackground": "#20201e",
    "editor.lineHighlightBorder": "#20201e00",
    "editor.selectionBackground": "#8cadff33",
    "editor.inactiveSelectionBackground": "#8cadff1a",
    "editor.selectionHighlightBackground": "#8cadff1a",
    "editor.wordHighlightBackground": "#8cadff1f",
    "editor.findMatchBackground": "#e3a94f55",
    "editor.findMatchHighlightBackground": "#e3a94f26",
    "editorLineNumber.foreground": "#4a4944",
    "editorLineNumber.activeForeground": "#a19d92",
    "editorIndentGuide.background1": "#262623",
    "editorIndentGuide.activeBackground1": "#3d3c37",
    "editorWhitespace.foreground": "#2e2d29",
    "editorBracketMatch.background": "#8cadff1f",
    "editorBracketMatch.border": "#8cadff80",
    "editorBracketHighlight.foreground1": "#a19d92",
    "editorBracketHighlight.foreground2": "#c9a3ee",
    "editorBracketHighlight.foreground3": "#7fd0d8",
    "editorGutter.background": "#1a1a18",
    "editorWidget.background": "#242421",
    "editorWidget.border": "#3d3c37",
    "editorSuggestWidget.background": "#242421",
    "editorSuggestWidget.border": "#3d3c37",
    "editorSuggestWidget.selectedBackground": "#2e2d29",
    "editorSuggestWidget.highlightForeground": "#8cadff",
    "editorHoverWidget.background": "#242421",
    "editorHoverWidget.border": "#3d3c37",
    "editorRuler.foreground": "#242421",
    focusBorder: "#8cadff",
    "scrollbar.shadow": "#00000000",
    "scrollbarSlider.background": "#3d3c3799",
    "scrollbarSlider.hoverBackground": "#4a494499",
    "scrollbarSlider.activeBackground": "#4a4944cc",
    "editorOverviewRuler.border": "#00000000",
  },
});

// Monaco measures its font once, when the first editor mounts. The web font
// may still be loading then, which leaves the cursor drifting off the text,
// so measure again once every font is in.
if (typeof document !== "undefined" && document.fonts) {
  void document.fonts.ready.then(() => monaco.editor.remeasureFonts());
}

loader.config({ monaco });

export const EDITOR_OPTIONS: monaco.editor.IStandaloneEditorConstructionOptions =
  {
    fontFamily: '"JetBrains Mono Variable", ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 13.5,
    lineHeight: 21,
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    smoothScrolling: true,
    cursorBlinking: "smooth",
    renderLineHighlight: "line",
    padding: { top: 14, bottom: 14 },
    tabSize: 4,
    insertSpaces: true,
    rulers: [88],
    bracketPairColorization: { enabled: true },
    automaticLayout: true,
    scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
    // The course is Python, and the reader is not writing a library here.
    quickSuggestions: { other: true, comments: false, strings: false },
    suggestOnTriggerCharacters: true,
    wordBasedSuggestions: "currentDocument",
  };

export { monaco };
