import { Suspense, memo, useMemo, useState, type ComponentProps, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { getDemo } from "../demos";
import { revealInEditor } from "../lib/events";
import rehypeLesson from "../lib/rehypeLesson";

const REMARK = [remarkGfm, remarkMath];

type Props = {
  children: string;
  /** Function names in the exercise. Inline code naming one becomes a link to it. */
  symbols?: string[];
  /** Renders one sentence, such as a flashcard's back, with no lesson structure. */
  inline?: boolean;
};

/**
 * Renders a lesson body: GitHub-flavoured Markdown, KaTeX for the maths, and
 * highlight.js for the code blocks, then `rehypeLesson` for the callouts,
 * sections, and highlights. A fenced ```demo block becomes an interactive
 * demo from `demos/index.ts`. Memoised because a lesson body is long and the
 * pane around it rerenders on every keystroke in the editor.
 */
function Markdown({ children, symbols, inline = false }: Props) {
  const rehype = useMemo(
    () =>
      [
        [rehypeKatex, { throwOnError: false, strict: false }],
        [rehypeHighlight, { detect: true, ignoreMissing: true }],
        ...(inline ? [] : [[rehypeLesson, { symbols: symbols ?? [] }]]),
      ] as never,
    [inline, symbols],
  );

  return (
    <div className={inline ? "lesson lesson-inline" : "lesson"}>
      <ReactMarkdown
        remarkPlugins={REMARK}
        rehypePlugins={rehype}
        components={{ pre: CodeBlock, code: InlineCode }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

export default memo(Markdown);

/** Inline code that names an exercise function opens it in the editor. */
function InlineCode({ node, ...props }: ComponentProps<"code"> & { node?: unknown }) {
  void node;
  const symbol = (props as Record<string, unknown>)["data-symbol"] as string | undefined;
  if (!symbol) return <code {...props} />;
  return (
    <button
      type="button"
      className="symbol"
      title={`Show ${symbol} in the editor`}
      onClick={() => revealInEditor(symbol)}
    >
      <code className={props.className}>{props.children}</code>
      <span className="symbol-arrow" aria-hidden>
        ↗
      </span>
    </button>
  );
}

/** The text inside a hast node, however deeply the highlighter nested it. */
function nodeText(node: unknown): string {
  if (!node || typeof node !== "object") return "";
  const candidate = node as { type?: string; value?: string; children?: unknown[] };
  if (candidate.type === "text") return candidate.value ?? "";
  if (Array.isArray(candidate.children)) return candidate.children.map(nodeText).join("");
  return "";
}

type DemoBlockJson = { id?: string; caption?: string; props?: Record<string, unknown> };

/**
 * One interactive demo, named by a fenced ```demo block.
 *
 * The block holds JSON with an `id`, a `caption`, and optionally `props`. An
 * id the registry does not know, or JSON that does not parse, renders as a
 * visible note rather than as nothing, because a silently missing figure is
 * the kind of thing nobody notices until a reader asks what the text means.
 */
function DemoBlock({ source }: { source: string }) {
  let parsed: DemoBlockJson;
  try {
    parsed = JSON.parse(source) as DemoBlockJson;
  } catch {
    return <MissingDemo reason="This demo block is not valid JSON." />;
  }
  const entry = parsed.id ? getDemo(parsed.id) : undefined;
  if (!entry) return <MissingDemo reason={`There is no demo called "${parsed.id ?? ""}".`} />;
  const { Component } = entry;
  return (
    <Suspense
      fallback={
        <div
          className="my-8 h-64 rounded-[10px] bg-chrome shadow-[inset_0_0_0_1px_var(--rule)]"
          aria-busy="true"
          aria-label="Loading the demo"
        />
      }
    >
      <Component {...(parsed.props ?? {})} caption={parsed.caption} />
    </Suspense>
  );
}

function MissingDemo({ reason }: { reason: string }) {
  return (
    <div className="callout callout-warning font-sans text-[13.5px]" role="note">
      {reason}
    </div>
  );
}

/** A code block, or a demo when the block is tagged `demo`. */
function CodeBlock(props: ComponentProps<"pre"> & { node?: unknown }) {
  const first = (props.node as { children?: unknown[] } | undefined)?.children?.[0] as
    | { type?: string; tagName?: string; properties?: Record<string, unknown> }
    | undefined;
  if (first?.type === "element" && first.tagName === "code") {
    const names = first.properties?.className;
    const list = Array.isArray(names) ? names.map(String) : [String(names ?? "")];
    if (list.includes("language-demo")) return <DemoBlock source={nodeText(first)} />;
  }
  return <PlainCodeBlock {...props} />;
}

/** A code block with its language named and a copy button. */
function PlainCodeBlock({ node, children, ...props }: ComponentProps<"pre"> & { node?: unknown }) {
  void node;
  const [copied, setCopied] = useState(false);
  const code = findCode(children);
  const language = /language-(\w+)/.exec(code?.className ?? "")?.[1];

  function copy() {
    const text = code?.text ?? "";
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1400);
      })
      .catch(() => undefined);
  }

  return (
    <div className="codeblock">
      <div className="codeblock-bar">
        <span>{language ?? "code"}</span>
        <button type="button" onClick={copy} aria-label="Copy the code">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre {...props}>{children}</pre>
    </div>
  );
}

function findCode(children: ReactNode): { className?: string; text: string } | null {
  const child = Array.isArray(children) ? children[0] : children;
  if (!child || typeof child !== "object" || !("props" in child)) return null;
  const props = child.props as { className?: string; children?: ReactNode };
  return { className: props.className, text: flatten(props.children) };
}

function flatten(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(flatten).join("");
  if (typeof node === "object" && "props" in node) {
    return flatten((node.props as { children?: ReactNode }).children);
  }
  return "";
}
