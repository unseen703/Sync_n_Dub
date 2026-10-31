import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from '../../lib/cn';
import { IconSpinner } from './Icons';

type Variant = 'primary' | 'ghost' | 'outline' | 'danger' | 'subtle';
type Size = 'sm' | 'md' | 'lg' | 'icon';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  busy?: boolean;
}

const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-brand-fill text-white shadow-lg shadow-brand/25 '
    + 'hover:bg-brand-soft active:translate-y-px',
  outline: 'border border-edge/70 bg-surface/70 text-ink2 hover:border-edge hover:bg-raised',
  ghost: 'text-ink3 hover:bg-ink/[0.07] hover:text-ink',
  subtle: 'bg-ink/[0.07] text-ink2 hover:bg-ink/[0.12] hover:text-ink',
  danger: 'border border-danger/45 bg-danger/10 text-danger hover:bg-danger/20',
};

const SIZES: Record<Size, string> = {
  sm: 'h-8 px-3 text-[13px] rounded-lg gap-1.5',
  md: 'h-9 px-3.5 text-[14px] rounded-xl gap-2',
  lg: 'h-11 px-5 text-[14px] rounded-xl gap-2',
  icon: 'h-9 w-9 rounded-lg justify-center',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'outline', size = 'md', busy, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={cn(
        'inline-flex items-center font-medium transition-colors duration-150',
        'disabled:cursor-not-allowed disabled:opacity-50',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {busy ? <IconSpinner className="shrink-0 opacity-85" /> : null}
      {children}
    </button>
  );
});

export function DemoChip({ title }: { title?: string }) {
  return (
    <span
      title={title ?? 'This endpoint is not built on the server yet — the response is a stub.'}
      className="rounded-md border border-warn/45 bg-warn/10 px-1.5 py-px
                 text-[10px] font-semibold uppercase tracking-wider text-warn"
    >
      demo
    </span>
  );
}
