import { AnimatePresence, m } from 'framer-motion';
import { useStore } from '../../store/useStore';
import { cn } from '../../lib/cn';
import { IconAlert, IconCheck, IconClose, IconInfo } from './Icons';

const TONE = {
  info: { box: 'border-edge/60 bg-raised', icon: 'text-brand-ink', Glyph: IconInfo },
  success: { box: 'border-ok/35 bg-raised', icon: 'text-ok', Glyph: IconCheck },
  warn: { box: 'border-warn/40 bg-raised', icon: 'text-warn', Glyph: IconAlert },
  error: { box: 'border-danger/45 bg-raised', icon: 'text-danger', Glyph: IconAlert },
} as const;

/**
 * Toasts sit bottom-RIGHT, opposite the job rail.
 *
 * The reasoning is the side the rail is on, not the side itself: a success
 * toast that covers the rail hides the very card whose state just changed.
 * The rail moved to the left on 2026-10-17 and this moved with it — if the
 * rail ever moves again, this follows.
 *
 * `role="status"` + `aria-live="polite"` on the region, so a screen reader
 * hears "The lip-synced render is ready" without the focus being stolen
 * mid-task.
 */
export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);

  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-[70] flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2"
      role="status"
      aria-live="polite"
    >
      <AnimatePresence initial={false}>
        {toasts.map((t) => {
          const tone = TONE[t.kind];
          const { Glyph } = tone;
          return (
            <m.div
              key={t.id}
              layout
              initial={{ opacity: 0, y: 12, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 6, scale: 0.98 }}
              transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
              className={cn(
                'pointer-events-auto flex gap-3 rounded-xl border px-3.5 py-3',
                'shadow-xl shadow-black/60',
                tone.box,
              )}
            >
              <Glyph width={15} height={15} className={cn('mt-0.5 shrink-0', tone.icon)} />
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-medium text-ink">{t.title}</div>
                {t.body ? (
                  <div className="mt-0.5 text-[12.5px] leading-relaxed text-ink3">{t.body}</div>
                ) : null}
                {t.action ? (
                  <button
                    type="button"
                    onClick={() => {
                      t.action?.run();
                      dismiss(t.id);
                    }}
                    className="mt-2 text-[12.5px] font-medium text-brand-ink underline-offset-2 hover:underline"
                  >
                    {t.action.label}
                  </button>
                ) : null}
              </div>
              <button
                type="button"
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss this notification"
                className="h-5 shrink-0 text-ink3 transition-colors hover:text-ink"
              >
                <IconClose width={14} height={14} />
              </button>
            </m.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
