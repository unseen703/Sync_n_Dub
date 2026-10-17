import { useMemo, useRef } from 'react';
import { useStore, filterJobs } from '../../store/useStore';
import { JobCard } from './JobCard';
import { Button } from '../ui/Button';
import { EmptyState, Skeleton } from '../ui/Card';
import { IconAlert, IconFilm, IconSearch } from '../ui/Icons';
import { normalizeState } from '../../lib/format';

/**
 * The job rail — the LEFT column, and the one you navigate from.
 *
 * Newest first, which is the order `_list_jobs` already returns
 * (`sorted(..., reverse=True)` over the timestamp-prefixed folder names).
 *
 * Arrow keys move between cards once one has focus, so a long list is
 * walkable without the mouse; `/` from anywhere focuses the search box.
 */
export function JobRail({ onNavigate }: { onNavigate?: () => void }) {
  const allJobs = useStore((s) => s.jobs);
  const loaded = useStore((s) => s.jobsLoaded);
  const error = useStore((s) => s.jobsError);
  const selected = useStore((s) => s.selected);
  const selectJob = useStore((s) => s.selectJob);
  const search = useStore((s) => s.search);
  const setSearch = useStore((s) => s.setSearch);
  const refreshJobs = useStore((s) => s.refreshJobs);
  const setNewJobOpen = useStore((s) => s.setNewJobOpen);
  // Only the selected job has live progress; every other card shows its
  // last known state. Polling 54 jobs to animate 53 idle bars is not worth
  // the requests.
  const livePct = useStore((s) => (s.progress?.job_id === s.selected ? s.progress?.pct : null));

  const listRef = useRef<HTMLDivElement>(null);
  const jobs = useMemo(() => filterJobs(allJobs, search), [allJobs, search]);
  const total = allJobs.length;
  const filtered = jobs.length !== total;
  const running = allJobs.filter((j) => {
    const s = normalizeState(j.state);
    return s === 'running' || s === 'queued';
  }).length;

  /** Roving focus with ↑/↓ once a card has it. */
  const onListKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    const cards = Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>('[data-job-card]') ?? [],
    );
    const i = cards.indexOf(document.activeElement as HTMLButtonElement);
    if (i < 0) return;
    e.preventDefault();
    cards[Math.max(0, Math.min(cards.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1)))]?.focus();
  };

  return (
    <div className="flex h-full w-full flex-col">
      <div className="flex items-center justify-between gap-2 px-4 pb-2.5 pt-4">
        <h2 className="text-[13px] font-semibold text-ink">Your dubs</h2>
        <span className="text-[11.5px] text-ink4">
          {loaded ? (running ? `${running} running` : `${total}`) : '…'}
        </span>
      </div>

      <div className="relative px-3 pb-2.5">
        <IconSearch
          width={14}
          height={14}
          className="pointer-events-none absolute left-[1.55rem] top-1/2 -translate-y-1/2 text-ink4"
        />
        <input
          id="job-search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              setSearch('');
              e.currentTarget.blur();
            }
          }}
          placeholder="Search by title or language"
          aria-label="Search your dubs"
          className="field h-9 pl-8"
          style={{ paddingLeft: '2rem' }}
        />
      </div>

      <div
        ref={listRef}
        onKeyDown={onListKeyDown}
        className="min-h-0 flex-1 space-y-1.5 overflow-y-auto px-3 pb-4"
      >
        {!loaded ? (
          <>
            <Skeleton className="h-[5.5rem]" />
            <Skeleton className="h-[5.5rem] opacity-70" />
            <Skeleton className="h-[5.5rem] opacity-40" />
          </>
        ) : error ? (
          <div className="rounded-xl border border-danger/40 bg-danger/[0.07] px-3 py-3">
            <div className="flex items-center gap-2 text-[12.5px] font-medium text-danger">
              <IconAlert width={14} height={14} />
              Could not load your dubs
            </div>
            <p className="mt-1 text-[12px] leading-relaxed text-ink3">
              The studio server did not answer. It may be restarting.
            </p>
            <Button size="sm" className="mt-2.5" onClick={() => void refreshJobs()}>
              Try again
            </Button>
          </div>
        ) : jobs.length === 0 && filtered ? (
          <div className="px-1 py-8 text-center">
            <p className="text-[12.5px] text-ink3">
              Nothing matches &ldquo;{search}&rdquo;.
            </p>
            <button
              type="button"
              onClick={() => setSearch('')}
              className="mt-1.5 text-[12.5px] font-medium text-brand-ink underline-offset-2 hover:underline"
            >
              Clear the search
            </button>
          </div>
        ) : jobs.length === 0 ? (
          <EmptyState
            className="px-2 py-10"
            icon={<IconFilm width={18} height={18} />}
            title="No dubs yet"
            body="Start one from a YouTube link and it will appear here."
            action={
              <Button variant="primary" size="sm" onClick={() => setNewJobOpen(true)}>
                New dub
              </Button>
            }
          />
        ) : (
          jobs.map((j) => (
            <JobCard
              key={j.job_id}
              job={j}
              active={j.job_id === selected}
              pct={j.job_id === selected ? livePct : null}
              onSelect={(id) => {
                selectJob(id);
                onNavigate?.();
              }}
            />
          ))
        )}
      </div>

      <div className="border-t border-line px-4 py-2.5 text-[11.5px] text-ink4">
        {loaded
          ? `${total} dub${total === 1 ? '' : 's'}${filtered ? ` · ${jobs.length} shown` : ''}`
          : 'Loading…'}
      </div>
    </div>
  );
}
