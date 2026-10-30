import { cn } from '../../lib/cn';

export function StagedValue({
  current,
  staged,
  placeholder = '—',
  className,

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
