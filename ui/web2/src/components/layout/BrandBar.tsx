import { useStore } from '../../store/useStore';
import { Button, DemoChip } from '../ui/Button';
import { IconKeyboard, IconMenu, IconMoon, IconPanel, IconPlus, IconSun } from '../ui/Icons';
import { Tooltip } from '../ui/Tooltip';
import { cn } from '../../lib/cn';
import { MOCKED } from '../../api';

/**
 * The product header: wordmark, what the product is, and the one action
 * that starts work.
 *
 * It spans the full width above all three columns rather than sitting in a
 * sidebar, because the thing it identifies is the product, not the job
 * list. The old UI put "Dubbing Studio" at the top of the left rail, which
 * made the brand read as a label for the list underneath it.
 */
export function BrandBar() {
  const setNewJobOpen = useStore((s) => s.setNewJobOpen);
  const toggleSpec = useStore((s) => s.toggleSpec);
  const specOpen = useStore((s) => s.specOpen);
  const setRailOpen = useStore((s) => s.setRailOpen);
  const railOpen = useStore((s) => s.railOpen);
  const theme = useStore((s) => s.theme);
  const toggleTheme = useStore((s) => s.toggleTheme);
  const selected = useStore((s) => s.selected);
  const nJobs = useStore((s) => s.jobs.length);
  const setShortcutsOpen = useStore((s) => s.setShortcutsOpen);

  return (
    <header
      className="flex h-[var(--brandbar)] flex-none items-center gap-2 border-b border-line
                 bg-surface/50 px-3 backdrop-blur-sm lg:px-4"
    >
      <a
        href="#/"
        className="group flex min-w-0 items-center gap-2.5"
        aria-label="Dubbing Studio — home"
      >
        <span
          className="grid h-7 w-7 flex-none place-items-center rounded-lg
                     bg-gradient-to-br from-brand to-aqua text-white shadow-lg shadow-brand/30"
          aria-hidden="true"
        >
          {/* A mouth/waveform mark: the two things this product joins. */}
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2.1" strokeLinecap="round">
            <path d="M5 10v4M9.5 6.5v11M14.5 8.5v7M19 11v2" />
          </svg>
        </span>
        <span className="truncate text-[14.5px] font-semibold tracking-tight text-ink">
          Dubbing Studio
        </span>
      </a>

      <span className="hidden text-[12.5px] text-ink3 sm:inline" aria-hidden="true">
        ·
      </span>
      <span className="hidden truncate text-[12.5px] text-ink3 sm:inline">
        Voice translation &amp; video editing
      </span>

      <div className="ml-auto flex items-center gap-1.5">
        <Tooltip label={specOpen ? 'Hide how this job was set up' : 'Show how this job was set up'}>
          <button
            type="button"
            onClick={toggleSpec}
            aria-pressed={specOpen}
            disabled={!selected}
            aria-label="Toggle the job settings panel"
            className={cn(
              'rounded-lg p-2 transition-colors disabled:opacity-40',
              specOpen
                ? 'bg-brand/20 text-brand-ink'
                : 'text-ink3 hover:bg-ink/[0.07] hover:text-ink',
            )}
          >
            <IconPanel />
          </button>
        </Tooltip>

        <Tooltip label="Keyboard shortcuts (?)">
          <button
            type="button"
            onClick={() => setShortcutsOpen(true)}
            aria-label="Keyboard shortcuts"
            className="hidden rounded-lg p-2 text-ink3 transition-colors hover:bg-ink/[0.07] hover:text-ink sm:block"
          >
            <IconKeyboard />
          </button>
        </Tooltip>

        <Button variant="primary" size="sm" onClick={() => setNewJobOpen(true)}>
          <IconPlus width={14} height={14} />
          New dub
          {MOCKED.newJob ? <DemoChip /> : null}
        </Button>

        <Tooltip label={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}>
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="rounded-lg p-2 text-ink3 transition-colors hover:bg-ink/[0.07] hover:text-ink"
          >
            {theme === 'dark' ? <IconSun /> : <IconMoon />}
          </button>
        </Tooltip>

        <Tooltip label={railOpen ? 'Hide the job list' : 'Show the job list'}>
          <button
            type="button"
            onClick={() => setRailOpen(!railOpen)}
            aria-label={`${railOpen ? 'Hide' : 'Show'} the job list (${nJobs} jobs)`}
            aria-pressed={railOpen}
            className={cn(
              'rounded-lg p-2 transition-colors hover:bg-ink/[0.07] hover:text-ink',
              railOpen ? 'text-ink' : 'text-ink3',
            )}
          >
            <IconMenu />
          </button>
        </Tooltip>
      </div>
    </header>
  );
}
