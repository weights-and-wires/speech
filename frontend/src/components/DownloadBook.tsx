import { useEffect, useState } from "react";

import { api, type BookFile } from "../lib/api";

/** Megabytes to one decimal place, which is all a download size needs. */
function size(bytes: number): string {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

/**
 * The whole course as one PDF book: every lesson, every exercise with its
 * starter file and hints, and the solutions in an appendix.
 *
 * The PDF is built at deploy time, so a server without one reports none and
 * the button does not appear.
 */
export default function DownloadBook() {
  const [book, setBook] = useState<BookFile | null>(null);

  useEffect(() => {
    api
      .book()
      .then((body) => setBook(body.book))
      .catch(() => undefined);
  }, []);

  if (!book) return null;

  const label = `Download the whole course as a PDF book (${book.pages} pages, ${size(book.bytes)})`;
  return (
    <a
      href={book.url}
      download
      aria-label={label}
      title={label}
      className="mr-1 flex min-h-10 items-center gap-1.5 rounded-md px-2 text-[13px] font-medium text-ink-3 transition-colors hover:bg-[color-mix(in_oklab,var(--ink)_6%,transparent)] hover:text-ink lg:min-h-8"
    >
      <DownloadIcon />
      <span className="hidden sm:inline">PDF</span>
    </a>
  );
}

function DownloadIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M8 2.5v7.5M4.8 7 8 10.2 11.2 7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 11.5v1.2c0 .5.4.8.8.8h8.4c.4 0 .8-.3.8-.8v-1.2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}
