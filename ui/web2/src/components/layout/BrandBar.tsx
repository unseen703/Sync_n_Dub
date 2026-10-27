import { useStore } from '../../store/useStore';
import { Button, DemoChip } from '../ui/Button';
import { IconKeyboard, IconMoon, IconPlus, IconSun } from '../ui/Icons';
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
  const theme = useStore((s) => s.theme);
  const toggleTheme = useStore((s) => s.toggleTheme);
  const setShortcutsOpen = useStore((s) => s.setShortcutsOpen);
  const mode = useStore((s) => s.mode);
  const toggleMode = useStore((s) => s.toggleMode);

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
        Indic video dubbing
      </span>

      <div className="ml-auto flex items-center gap-1.5">
        {/*
          A view switch, not a permission. It only changes how much of the
          pipeline the screen admits to; every control stays reachable in
          both modes. See `mode` in the store for why it is not a check.
        */}
        <Tooltip
          label={
            mode === 'admin'
              ? 'Showing technical detail — switch to the simple view'
              : 'Show technical detail: run config, model names, job ids, errors'
          }
        >
          <div
            role="group"
            aria-label="View mode"
            className="hidden rounded-lg border border-line bg-canvas/60 p-0.5 sm:flex"
          >
            {(['user', 'admin'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => { if (mode !== m) toggleMode(); }}
                aria-pressed={mode === m}
                className={cn(
                  'rounded-[6px] px-2 py-1 text-[11.5px] font-medium capitalize transition-colors',
                  mode === m
                    ? 'bg-brand/20 text-brand-ink'
                    : 'text-ink3 hover:text-ink',
                )}
              >
                {m}
              </button>
            ))}
          </div>
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

      </div>
    </header>
  );
}
