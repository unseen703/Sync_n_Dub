import { AnimatePresence, m } from 'framer-motion';
import { BrandBar } from './BrandBar';
import { JobRail } from './JobRail';
import { Stage } from './Stage';
import { Toasts } from '../ui/Toasts';
import { Tooltip } from '../ui/Tooltip';
import { IconMenu } from '../ui/Icons';
import { useStore } from '../../store/useStore';

/**
 * Brand bar across the top, then three columns:
 *
 *     ┌──────────────── Dubbing Studio ────────────────┐
 *     │    jobs    │        the dub        │  details  │
 *     └────────────┴───────────────────────┴───────────┘
 *
 * **Jobs on the LEFT, this job's details on the RIGHT.** Navigation is the
 * column you scan down to pick something, and it goes where a reader starts.
 * The details panel is reference about whatever is already selected, so it
 * trails the thing it describes rather than preceding it.
 *
 * The two panels are separate, not one stacked sidebar, because they have
 * different lifetimes: the job list is navigation and is wanted constantly,
 * while the details are read once per job and then become dead width next
 * to the video. Both can be closed; the list defaults open, the details
 * remember whatever the user last chose.
 *
 * Responsive behaviour:
 *   < 1024 px   both are off-canvas drawers — the job list flies in from
 *               the LEFT and the details from the RIGHT, each matching the
 *               side it occupies at desktop width.
 *   >= 1024 px  the job list is inline and collapsible.
 *   >= 1536 px  the details PUSH instead of overlaying, where there is room
 *               without squeezing the video below a useful 16:9.
 */
export function AppShell() {
  const railOpen = useStore((s) => s.railOpen);
  const setRailOpen = useStore((s) => s.setRailOpen);
  // The job list is inline from 1024 px up and a drawer below it. `railOpen`
  // drives both, so the toggle in the brand bar means the same thing at
  // every width.
  const railInline = railOpen;
  const railDrawer = railOpen;

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden">
      <BrandBar />

      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* Job list — inline on the LEFT at >=1024 px, collapsible. */}
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

        {/* With the collapse control now INSIDE the rail, a closed rail
            would have no way back — so the reopen affordance lives here,
            as a thin strip on the edge the rail flies out from. */}
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

        {/* Job list, off-canvas from the LEFT below 1024 px. */}
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
