/**
 * The course mark: two inputs feeding one unit, on a rounded tile. It appears
 * in the header and on the sign-in page, so it lives in one place. The
 * favicon in index.html draws the same shape.
 */
export default function Mark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect width="24" height="24" rx="6" fill="var(--primary)" />
      <g fill="var(--on-primary)">
        <circle cx="6.5" cy="7.5" r="2" />
        <circle cx="6.5" cy="16.5" r="2" />
        <circle cx="17.5" cy="12" r="2.2" />
      </g>
      <path
        d="M8.3 8.3 15.6 11.3M8.3 15.7 15.6 12.7"
        stroke="var(--on-primary)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
