import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '../../lib/cn';

export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('card', className)} {...rest} />;
}

/** Small-caps section label. 11px / #7E8CA0 — 5.7:1 on the canvas. */
export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('eyebrow', className)}>{children}</div>;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('sk', className)} aria-hidden="true" />;
}

/**
 * The one empty state. Every panel that can be empty uses it, so "nothing
 * here" always reads the same way and always says what to do about it — a
 * blank area is indistinguishable from a bug.
 */
export function EmptyState({
  icon,
  title,
  body,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  body?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 px-6 py-14 text-center',
        className,
      )}
    >
      <div
        className="grid h-11 w-11 place-items-center rounded-2xl border border-line
                   bg-gradient-to-br from-brand/25 to-aqua/15 text-brand-ink"
        aria-hidden="true"
      >
        {icon}
      </div>
      <div className="t-title">{title}</div>
      {body ? <p className="max-w-sm text-[13px] leading-relaxed text-ink3">{body}</p> : null}
      {action}
    </div>
  );
}
