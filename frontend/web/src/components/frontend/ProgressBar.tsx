import { cn } from '../../lib/cn';

export function ProgressBar({
  pct,
  className,
  tone = 'accent',
  indeterminate = false,
}: {
  pct: number;
  className?: string;
  tone?: 'accent' | 'emerald' | 'rose' | 'slate';
  indeterminate?: boolean;
}) {
  const fill = {
    accent: 'from-brand to-aqua',
    emerald: 'from-ok/80 to-ok',
    rose: 'from-danger-fill to-danger',
    slate: 'from-edge to-ink4',
  }[tone];
  const v = Math.max(0, Math.min(100, Math.round(pct)));
  return (
    <div
      className={cn('h-1.5 w-full overflow-hidden rounded-full bg-ink/[0.09]', className)}
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : v}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={cn(
          'h-full rounded-full bg-gradient-to-r transition-[width] duration-500 ease-out',
          fill,
          indeterminate && 'w-1/3 animate-pulse-soft',
        )}
        style={indeterminate ? undefined : { width: `${v}%` }}
      />
    </div>
  );
}
