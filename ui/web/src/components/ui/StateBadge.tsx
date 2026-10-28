import { cn } from '../../lib/cn';
import { normalizeState } from '../../lib/format';

/**
 * One definition of what each job state looks like and is called.
 *
 * Every text colour here is a palette ink token, so each measures at least
 * 9:1 against its own tinted pill — the old badges used `text-slate-300`
 * class colours against 10 %-opacity fills and several landed near 3:1.
 */
const STYLES: Record<string, string> = {
  done: 'border-ok/35 bg-ok/10 text-ok',
  running: 'border-brand-ink/40 bg-brand/15 text-brand-ink',
  queued: 'border-edge/60 bg-ink/[0.06] text-ink2',
  failed: 'border-danger/45 bg-danger/10 text-danger',
  cancelled: 'border-warn/40 bg-warn/10 text-warn',
  skipped: 'border-line bg-ink/[0.04] text-ink4',
  pending: 'border-line bg-ink/[0.04] text-ink4',
  unknown: 'border-line bg-ink/[0.04] text-ink3',
};

const WORDS: Record<string, string> = {
  done: 'Done',
  running: 'Running',
  queued: 'Queued',
  failed: 'Failed',
  cancelled: 'Stopped',
  skipped: 'Skipped',
  pending: 'Waiting',
  unknown: 'Unknown',
};

export function stateStyle(raw: string | null | undefined): string {
  return STYLES[normalizeState(raw)] ?? STYLES.unknown;
}

export function stateWord(raw: string | null | undefined): string {
  return WORDS[normalizeState(raw)] ?? 'Unknown';
}

export function StateBadge({
  state,
  className,
  dot = true,
}: {
  state: string | null | undefined;
  className?: string;
  dot?: boolean;
}) {
  const s = normalizeState(state);
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5',
        'text-[11.5px] font-medium',
        stateStyle(s),
        className,
      )}
    >
      {dot ? (
        <span
          className={cn(
            'h-1.5 w-1.5 rounded-full bg-current',
            s === 'running' && 'animate-pulse-soft',
          )}
        />
      ) : null}
      {stateWord(s)}
    </span>
  );
}
