import * as RD from '@radix-ui/react-dialog';
import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';
import { IconClose } from './Icons';

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  width = 'max-w-lg',
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  width?: string;
}) {
  return (
    <RD.Root open={open} onOpenChange={onOpenChange}>
      <RD.Portal>
        <RD.Overlay className="fixed inset-0 z-50 animate-fade-in bg-black/70 backdrop-blur-sm" />
        <RD.Content
          className={cn(
            'fixed left-1/2 top-1/2 z-50 w-[calc(100vw-2rem)] -translate-x-1/2 -translate-y-1/2',
            'animate-slide-up rounded-2xl border border-line bg-raised shadow-2xl shadow-black/70',
            'flex max-h-[88vh] flex-col overflow-hidden',
            width,
          )}
        >
          <div className="flex items-start gap-3 border-b border-line px-5 py-4">
            <div className="min-w-0 flex-1">
              <RD.Title className="t-title">{title}</RD.Title>
              {description ? (
                <RD.Description className="mt-1 text-[12.5px] leading-relaxed text-ink3">
                  {description}
                </RD.Description>
              ) : null}
            </div>
            <RD.Close
              className="rounded-lg p-1.5 text-ink3 transition-colors hover:bg-ink/[0.08] hover:text-ink"
              aria-label="Close"
            >
              <IconClose />
            </RD.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
          {footer ? (
            <div className="flex items-center justify-end gap-2 border-t border-line bg-canvas/50 px-5 py-3.5">
              {footer}
            </div>
          ) : null}
        </RD.Content>
      </RD.Portal>
    </RD.Root>
  );
}

export const DialogClose = RD.Close;
