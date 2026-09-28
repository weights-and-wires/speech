import { createRoot } from "react-dom/client";

import Book, { type BookData } from "./Book";
import "../index.css";
import "./book.css";

/**
 * The print layout's entry point. scripts/build-book.mjs serves the lessons
 * and exercises as `book.json` beside this page, opens it, and prints it.
 */
async function start() {
  const data = (await (await fetch("book.json")).json()) as BookData;
  createRoot(document.getElementById("book")!).render(<Book data={data} />);
}

start().catch((error: unknown) => {
  window.__book = { error: String(error) };
});
