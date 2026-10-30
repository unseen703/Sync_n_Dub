import { lazy, Suspense, useState } from 'react';
import { AnimatePresence, m } from 'framer-motion';
import { useStore, selectStagedCount } from '../../store/useStore';
import { Button, DemoChip } from '../ui/Button';
import { MOCKED } from '../../api';

const ApplyDialog = lazy(() =>
  import('./ApplyDialog').then((m) => ({ default: m.ApplyDialog })),
);

export function PendingBar() {
  const count = useStore(selectStagedCount);
  const discard = useStore((s) => s.discardStaged);
  const [open, setOpen] = useState(false);

  return (
    <>
      <AnimatePresence>
        {count > 0 ? (
          <m.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
            className="sticky bottom-3 z-30 mt-2 flex flex-wrap items-center gap-3 rounded-xl
                       border border-warn/45 bg-raised px-3.5 py-3 shadow-xl shadow-black/60"
          >
            <span className="flex items-center gap-2 text-[13px] text-ink">
              <span className="h-1.5 w-1.5 rounded-full bg-warn" aria-hidden="true" />
              {count} change{count === 1 ? '' : 's'} not applied yet
            </span>
            <span className="hidden text-[12px] text-ink3 sm:inline">
              Nothing is re-rendered until you apply them.
            </span>
            <div className="ml-auto flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => void discard()}>
                Discard
              </Button>
              <Button variant="primary" size="sm" onClick={() => setOpen(true)}>
                Review &amp; apply
                {MOCKED.edits ? <DemoChip /> : null}
              </Button>
            </div>
          </m.div>
        ) : null}
      </AnimatePresence>

      {open ? (
        <Suspense fallback={null}>
          <ApplyDialog open={open} onOpenChange={setOpen} />
        </Suspense>
      ) : null}
    </>
  );
}
