import { AnimatePresence, m } from 'framer-motion';
import { BrandBar } from './BrandBar';
import { JobRail } from './JobRail';
import { Stage } from './Stage';
import { Toasts } from '../frontend/Toasts';
import { Tooltip } from '../frontend/Tooltip';
import { IconMenu } from '../frontend/Icons';
import { useStore } from '../../store/useStore';

/** Main responsive layout. It reads the selected job from the store and renders navigation plus the active workspace. */
export function AppShell() {
  const railOpen = useStore((s) => s.railOpen);
  const setRailOpen = useStore((s) => s.setRailOpen);
  const selected = useStore((s) => s.selected);
  const talkingHead = useStore((s) => s.talkingHead);

  const railInline = railOpen;
  const railDrawer = railOpen;

  const isLanding = !selected && !talkingHead;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden">
      <BrandBar showNewDub={!isLanding} />

      <div className="flex min-h-0 flex-1 overflow-hidden">

        <AnimatePresence initial={false}>
          {railInline ? (
            <m.aside
              key="rail-inline"
              className="hidden shrink-0 overflow-hidden border-r border-line
                         bg-canvas/40 lg:block"
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 'var(--rail)', opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              aria-label="Jobs"
            >
              <div className="h-full w-[var(--rail)]">
                <JobRail />
              </div>
            </m.aside>
          ) : null}
        </AnimatePresence>

        {!railOpen ? (
          <div className="hidden shrink-0 border-r border-line bg-canvas/40 lg:block">
            <Tooltip label="Show the job list" side="right">
              <button
                type="button"
                onClick={() => setRailOpen(true)}
                aria-label="Show the job list"
                aria-expanded={false}
                aria-controls="job-rail-list"
                className="m-1.5 rounded-lg p-2 text-ink3 transition-colors
                           hover:bg-ink/[0.07] hover:text-ink"
              >
                <IconMenu />
              </button>
            </Tooltip>
          </div>
        ) : null}

        <AnimatePresence>
          {railDrawer ? (
            <>
              <m.div
                className="fixed inset-0 z-40 bg-black/70 lg:hidden"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                onClick={() => setRailOpen(false)}
              />
              <m.aside
                className="fixed inset-y-0 left-0 z-50 flex w-[var(--rail)] max-w-[85vw]
                           border-r border-line bg-surface shadow-2xl shadow-black/70 lg:hidden"
                initial={{ x: '-100%' }}
                animate={{ x: 0 }}
                exit={{ x: '-100%' }}
                transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                aria-label="Jobs"
              >
                <JobRail onNavigate={() => setRailOpen(false)} />
              </m.aside>
            </>
          ) : null}
        </AnimatePresence>

        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <Stage />
        </main>
      </div>

      <Toasts />
    </div>
  );
}
