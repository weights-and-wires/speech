import { Suspense } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { DEMOS } from "../demos";
import { PrintedFigure } from "../demos/kit";

const IDS = Object.keys(DEMOS);
/** The book's text block, 138 mm at 96 px to the inch, as src/book/book.css sets it. */
const BOOK_WIDTH_PX = Math.floor((138 / 25.4) * 96);

/**
 * Every registered demo on one page, so a demo can be seen and checked
 * without the lesson that uses it. Only the dashboard's footer links here.
 * `?printed=1` shows a demo the way the PDF book prints it: at the book's
 * width, inside `PrintedFigure`, with its controls hidden.
 */
export default function DemoGallery() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [search, setSearch] = useSearchParams();
  const printed = search.get("printed") === "1";
  const suffix = printed ? "?printed=1" : "";
  const entry = id ? DEMOS[id] : undefined;
  const at = id ? IDS.indexOf(id) : -1;
  const previous = at > 0 ? IDS[at - 1] : null;
  const next = at >= 0 && at < IDS.length - 1 ? IDS[at + 1] : null;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="mx-auto max-w-3xl px-5 pb-16 pt-8 sm:px-8 lg:pt-10">
        <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2">
          <h1 className="font-serif text-[1.6rem] font-semibold leading-tight text-ink">
            <Link to="/demos" className="hover:text-accent">
              Demos
            </Link>
          </h1>
          <span className="text-[12.5px] tabular-nums text-ink-4">{IDS.length} registered</span>
        </header>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <label htmlFor="demo-picker" className="sr-only">
            Choose a demo
          </label>
          <select
            id="demo-picker"
            value={entry ? id : ""}
            onChange={(event) => navigate(event.target.value ? `/demos/${event.target.value}${suffix}` : "/demos")}
            className="min-h-9 min-w-0 flex-1 rounded-md bg-raise px-2 text-[13.5px] text-ink shadow-[inset_0_0_0_1px_var(--rule-strong)]"
          >
            <option value="">All demos</option>
            {IDS.map((demoId) => (
              <option key={demoId} value={demoId}>
                {demoId}
              </option>
            ))}
          </select>
          {entry && (
            <>
              <Link
                to={previous ? `/demos/${previous}${suffix}` : "#"}
                aria-disabled={!previous}
                className={`btn btn-secondary min-h-9! text-[12.5px] ${previous ? "" : "pointer-events-none opacity-40"}`}
              >
                Previous
              </Link>
              <Link
                to={next ? `/demos/${next}${suffix}` : "#"}
                aria-disabled={!next}
                className={`btn btn-secondary min-h-9! text-[12.5px] ${next ? "" : "pointer-events-none opacity-40"}`}
              >
                Next
              </Link>
            </>
          )}
        </div>

        {id && !entry && (
          <p className="mt-8 text-[14px] text-ink-3">There is no demo called “{id}”.</p>
        )}

        {entry && (
          <label className="mt-3 flex w-fit cursor-pointer select-none items-center gap-2 text-[12.5px] text-ink-3">
            <input
              type="checkbox"
              checked={printed}
              onChange={(event) => setSearch(event.target.checked ? { printed: "1" } : {})}
              className="size-3.5 cursor-pointer accent-[var(--demo)]"
            />
            As the book prints it
          </label>
        )}

        {entry && id ? (
          <div className="mt-2" style={printed ? { maxWidth: BOOK_WIDTH_PX } : undefined}>
            <Suspense
              fallback={
                <div className="my-8 h-64 animate-pulse rounded-[10px] bg-chrome" aria-busy="true" aria-label="Loading the demo" />
              }
            >
              {printed ? (
                <PrintedFigure.Provider value={{ url: `${window.location.host}/demos/${id}` }}>
                  <entry.Component key={`${id}-printed`} caption="Shown in the gallery, outside its lesson." />
                </PrintedFigure.Provider>
              ) : (
                <entry.Component key={id} caption="Shown in the gallery, outside its lesson." />
              )}
            </Suspense>
          </div>
        ) : (
          !id && (
            <ol className="mt-6">
              {IDS.map((demoId, index) => (
                <li key={demoId} className="border-b border-rule">
                  <Link
                    to={`/demos/${demoId}`}
                    className="group -mx-3 flex items-baseline gap-3 rounded-md px-3 py-3 transition-colors hover:bg-[color-mix(in_oklab,var(--ink)_4%,transparent)]"
                  >
                    <span className="w-6 shrink-0 text-[13px] tabular-nums text-ink-4">{index + 1}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-[14.5px] font-medium leading-snug text-ink group-hover:text-accent">
                        {DEMOS[demoId].title}
                      </span>
                      <span className="mt-0.5 block break-all font-mono text-[12px] text-ink-4">{demoId}</span>
                    </span>
                  </Link>
                </li>
              ))}
            </ol>
          )
        )}
      </div>
    </div>
  );
}
