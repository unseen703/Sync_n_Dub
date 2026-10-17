import * as RT from '@radix-ui/react-tooltip';
import type { ReactNode } from 'react';

export const TooltipProvider = RT.Provider;

export function Tooltip({
  label,
  children,
  side = 'top',
}: {
  label: ReactNode;
  children: ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
}) {
  if (!label) return <>{children}</>;
  return (
    <RT.Root>
      <RT.Trigger asChild>{children}</RT.Trigger>
      <RT.Portal>
        <RT.Content
          side={side}
          sideOffset={6}
          collisionPadding={8}
          className="z-50 max-w-xs animate-fade-in rounded-lg border border-line
                     bg-raised px-2.5 py-1.5 text-[12px] leading-relaxed text-ink2
                     shadow-xl shadow-black/60"
        >
          {label}
          <RT.Arrow className="fill-raised" width={10} height={5} />
        </RT.Content>
      </RT.Portal>
    </RT.Root>
  );
}
