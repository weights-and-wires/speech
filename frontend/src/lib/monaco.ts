/**
 * Monaco, bundled with the app instead of fetched from a CDN at run time.
 *
 * `@monaco-editor/react` loads the editor from jsdelivr unless it is handed an
 * instance. The app is behind a login on a private host, so it builds a lean
 * instance with the features and the one language the exercises need, gives it
 * the code-slab themes, and registers it with the loader.
 */

import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/editor/editor.api";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";

import { defineSlabThemes } from "../ui/monaco-theme";

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

// The editor is the dark-green code slab in both page themes, the same as
// the lesson code blocks. The themes come from the shared design system in
// src/ui, under the names the editor has always used.
export const THEMES = defineSlabThemes(monaco.editor, { light: "ww-light", dark: "ww-dark" });

// Monaco measures its font once, when the first editor mounts. Measure again
// once every font is in, so the cursor never drifts off the text.
if (typeof document !== "undefined" && document.fonts) {
  void document.fonts.ready.then(() => monaco.editor.remeasureFonts());
}

loader.config({ monaco });

export const EDITOR_OPTIONS: monaco.editor.IStandaloneEditorConstructionOptions =
  {
    // The --mono stack from src/ui/type.css. Monaco needs the names, not
    // the variable.
    fontFamily: '"SFMono-Regular", ui-monospace, Consolas, "Liberation Mono", monospace',
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
