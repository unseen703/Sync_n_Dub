import { useStore } from '../../store/useStore';
import { Button, DemoChip } from '../frontend/Button';
import { IconKeyboard, IconMoon, IconPlus, IconSun } from '../frontend/Icons';
import { Tooltip } from '../frontend/Tooltip';
import { MOCKED } from '../../api';

export function BrandBar({ showNewDub = true }: { showNewDub?: boolean }) {
  const setNewJobOpen = useStore((s) => s.setNewJobOpen);
  const theme = useStore((s) => s.theme);
  const toggleTheme = useStore((s) => s.toggleTheme);
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

        {showNewDub ? (
          <Button variant="primary" size="sm" onClick={() => setNewJobOpen(true)}>
            <IconPlus width={14} height={14} />
            New dub
            {MOCKED.newJob ? <DemoChip /> : null}
          </Button>
        ) : null}

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
