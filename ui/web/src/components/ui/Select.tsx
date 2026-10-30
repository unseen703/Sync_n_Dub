import * as RS from '@radix-ui/react-select';
import { Fragment, type ReactNode } from 'react';
import { cn } from '../../lib/cn';
import { IconCheck, IconChevron } from './Icons';

export interface SelectItem {
  value: string;
  label: ReactNode;
  hint?: string;
  disabled?: boolean;

  group?: string;
}

export function Select({
  value,
  onValueChange,
  items,
  placeholder = 'Choose…',
  className,
  contentClassName,
  ariaLabel,
  disabled,
}: {
  value: string | undefined;
  onValueChange: (v: string) => void;
  items: SelectItem[];
  placeholder?: string;
  className?: string;
  contentClassName?: string;
  ariaLabel?: string;
  disabled?: boolean;
}) {
  let lastGroup: string | undefined;

  return (
    <RS.Root value={value} onValueChange={onValueChange} disabled={disabled}>
      <RS.Trigger
        aria-label={ariaLabel}
        className={cn(
          'inline-flex h-9 w-full items-center justify-between gap-2 rounded-xl',
          'border border-edge/70 bg-surface/70 px-3 text-[13px] text-ink',
          'transition-colors hover:border-edge data-[placeholder]:text-ink4',
          'disabled:cursor-not-allowed disabled:opacity-50',
          className,
        )}
      >
        <span className="truncate text-left">
          <RS.Value placeholder={placeholder} />
        </span>
        <RS.Icon className="shrink-0">
          <IconChevron className="text-ink3" />
        </RS.Icon>
      </RS.Trigger>
      <RS.Portal>
        <RS.Content
          position="popper"
          sideOffset={6}
          collisionPadding={12}
          className={cn(
            'z-50 max-h-80 min-w-[var(--radix-select-trigger-width)] overflow-hidden',
            'rounded-xl border border-line bg-raised shadow-2xl shadow-black/70',
            contentClassName,
          )}
        >
          <RS.Viewport className="p-1.5">
            {items.map((it) => {
              const heading = it.group && it.group !== lastGroup ? it.group : null;
              lastGroup = it.group;
              return (
                <Fragment key={it.value}>
                  {heading ? (
                    <div className="eyebrow px-2 pb-1 pt-2 first:pt-0">{heading}</div>
                  ) : null}
                  <RS.Item
                    value={it.value}
                    disabled={it.disabled}
                    className="relative flex cursor-pointer select-none items-center gap-2 rounded-lg
                               px-2 py-1.5 pr-8 text-[13px] text-ink2 outline-none
                               data-[highlighted]:bg-brand/25 data-[highlighted]:text-ink
                               data-[disabled]:opacity-45"
                  >
                    <div className="min-w-0 flex-1">
                      <RS.ItemText>{it.label}</RS.ItemText>
                      {it.hint ? (
                        <div className="truncate text-[11.5px] text-ink4">{it.hint}</div>
                      ) : null}
                    </div>
                    <RS.ItemIndicator className="absolute right-2">
                      <IconCheck className="text-aqua-ink" width={14} height={14} />
                    </RS.ItemIndicator>
                  </RS.Item>
                </Fragment>
              );
            })}
          </RS.Viewport>
        </RS.Content>
      </RS.Portal>
    </RS.Root>
  );
}
