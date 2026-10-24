import { cn } from '../../lib/cn';

/**
 * One value, showing what it would become.
 *
 * `current` alone when nothing is staged; `current` struck through, an arrow,
 * then the staged value in blue when something is. Three places need this —
 * the cast rows (gender, voice, tone), the script list, and the agent's own
 * summary lines — and they used to each draw their own version: a `was X`
 * suffix here, a truncated `was X` there, a strikethrough somewhere else. The
 * same fact rendered three ways reads as three different kinds of fact.
 *
 * `blue-ink` rather than `brand-ink` is deliberate. Brand means "active" or
 * "selected" everywhere else in this app, and a staged edit is the opposite
 * claim: it is explicitly NOT yet true, and will not be until Apply spends
 * the GPU. Giving it the selected colour made a pending change look like a
 * current state.
 *
 * `staged === current` renders as unstaged. An edit that changes nothing is
 * not worth an arrow, and the agent can produce one — asking for a voice a
 * speaker already has parses fine and stages an op that re-renders the dub
 * into an identical result.
 */
export function StagedValue({
  current,
  staged,
  placeholder = '—',
  className,
  /** Shown to a screen reader in place of the visual strikethrough+arrow. */
  label,
}: {
  current: string | null | undefined;
  staged?: string | null;
  placeholder?: string;
  className?: string;
  label?: string;
}) {
  const now = current ?? null;
  const next = staged ?? null;

  if (next === null || next === now) {
    return <span className={cn('text-ink', className)}>{now ?? placeholder}</span>;
  }

  return (
    <span className={className}>
      {/* The strikethrough and the arrow are decoration; the sentence below
          is what a screen reader gets, because "Amit arrow Kavya" is not a
          thing anyone can act on. */}
      <span aria-hidden="true">
        <span className="text-ink4 line-through">{now ?? placeholder}</span>
        <span className="mx-1 text-ink4">→</span>
        <span className="font-medium text-blue-ink">{next}</span>
      </span>
      <span className="sr-only">
        {label ? `${label}: ` : ''}
        {now ?? placeholder}, staged to change to {next}
      </span>
    </span>
  );
}

/**
 * The same before/after over a block of TEXT rather than a single value.
 *
 * Used for a rewritten line, where the two readings are sentences and have
 * to sit on their own rows to be comparable. The old wording is shown in
 * FULL: the script list used to truncate it to 24 characters, which is
 * shorter than most cues and so hid the very words that were changing.
 */
export function StagedText({
  current,
  staged,
  note,
}: {
  current: string;
  staged: string;
  note?: string;
}) {
  return (
    <div className="mt-1 space-y-0.5 border-l-2 border-blue-ink/40 pl-2">
      <div className="text-[12px] leading-snug text-ink4 line-through">{current}</div>
      <div className="text-[12.5px] font-medium leading-snug text-blue-ink">{staged}</div>
      {note ? <div className="text-[11px] text-ink4">{note}</div> : null}
    </div>
  );
}
