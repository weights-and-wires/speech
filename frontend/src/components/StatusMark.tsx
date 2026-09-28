/**
 * A lesson's state as a small glyph: a tick once solved, a half-filled circle
 * once started, and an empty circle before that. The sidebar and the contents
 * page share it, so the two never disagree about what a state looks like.
 */
export default function StatusMark({
  state,
  className = "",
}: {
  state?: "started" | "solved";
  className?: string;
}) {
  const label = state === "solved" ? "Solved" : state === "started" ? "In progress" : "Not started";
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      role="img"
      aria-label={label}
      className={`shrink-0 ${className}`}
    >
      <title>{label}</title>
      {state === "solved" ? (
        <>
          <circle cx="7" cy="7" r="6.5" fill="var(--good)" />
          <path d="M4.2 7.2 6.1 9 9.8 5.2" stroke="var(--canvas)" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </>
      ) : state === "started" ? (
        <>
          <circle cx="7" cy="7" r="5.75" stroke="var(--ink-3)" strokeWidth="1.3" fill="none" />
          <path d="M7 3.25a3.75 3.75 0 0 1 0 7.5z" fill="var(--ink-3)" />
        </>
      ) : (
        <circle cx="7" cy="7" r="5.75" stroke="var(--rule-strong)" strokeWidth="1.3" fill="none" />
      )}
    </svg>
  );
}
