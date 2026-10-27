import { memo } from 'react';
import { cn } from '../../lib/cn';
import { fmtRelative, jobTitle, jobWhen, normalizeState } from '../../lib/format';
import { describeLang, langClass } from '../../lib/lang';
import { ProgressBar } from '../ui/ProgressBar';
import type { JobRow } from '../../api/types';

/**
 * One card in the rail.
 *
 * The title is the video title. `GET /api/jobs` does not carry one yet
 * (`_list_jobs` never reads `meta.json`), so `jobTitle()` falls back
 * title -> slug -> job_id and flags the fallback. A degraded name is
 * rendered in mono and dimmed so a missing title LOOKS missing instead of
 * passing as the name of the video.
 *
 * A failed job says so, in words, directly under the title.
 */
export const JobCard = memo(function JobCard({
  job,
  active,
  pct,
  onSelect,
}: {
  job: JobRow;
  active: boolean;
  pct?: number | null;
  onSelect: (id: string) => void;
}) {
  const state = normalizeState(job.state);
  const { text, degraded } = jobTitle(job);
  const failed = state === 'failed';
  const running = state === 'running';

  return (
    <button
      type="button"
      data-job-card
      onClick={() => onSelect(job.job_id)}
      aria-current={active ? 'true' : undefined}
      className={cn(
        'group relative w-full overflow-hidden rounded-xl border px-3 py-2.5 text-left',
        'transition-colors duration-150',
        active
          ? 'border-brand-ink/50 bg-brand/[0.16]'
          : 'border-line bg-surface/50 hover:border-edge/70 hover:bg-surface',
      )}
    >
      <span
        className={cn(
          'absolute inset-y-2 left-0 w-[3px] rounded-full transition-opacity',
          active ? 'bg-gradient-to-b from-brand to-aqua opacity-100' : 'opacity-0',
        )}
        aria-hidden="true"
      />

      <div
        className={cn(
          'truncate text-[13px] font-medium leading-snug',
          degraded ? 'font-mono text-[11.5px] text-ink3' : 'text-ink',
        )}
        title={degraded ? `No title was recorded for ${job.job_id}` : text}
      >
        {text}
      </div>

      <div className="mt-1 flex items-center gap-1.5 text-[11.5px]">
        {failed ? (
          <span className="flex items-center gap-1.5 font-medium text-danger">
            <span className="h-1.5 w-1.5 rounded-full bg-danger" aria-hidden="true" />
            Failed
          </span>
        ) : (
          <>
            {job.language ? (
              <span className={cn('text-ink3', langClass(job.language))}>
                {describeLang(job.language).split(' (')[0]}
              </span>
            ) : (
              <span className="text-ink4">Language not recorded</span>
            )}
            <span className="text-ink4" aria-hidden="true">
              ·
            </span>
            {state === 'done' ? (
              <span className="text-ok">Ready</span>
            ) : state === 'queued' ? (
              <span className="text-ink3">Queued</span>
            ) : running ? (
              <span className="text-brand-ink">Running</span>
            ) : (
              <span className="text-ink4">{fmtRelative(job.updated)}</span>
            )}
          </>
        )}
      </div>

      {/*
        When and what. Several jobs can share a title — six runs of the same
        video is the normal way of working here — so without a timestamp the
        cards are indistinguishable and clicking one is a guess. `timestamp`
        comes from the job id, so it is present even for a job with no
        status file at all.
      */}
      <div className="mt-1 flex items-center gap-1.5 text-[11px] text-ink4">
        <span className="tabular-nums">
          {job.timestamp ? jobWhen(job.timestamp) : fmtRelative(job.updated)}
        </span>
        {job.lipsync_model ? (
          <>
            <span aria-hidden="true">·</span>
            <span className="truncate">{job.lipsync_model}</span>
          </>
        ) : null}
      </div>

      {running ? (
        <div className="mt-2 flex items-center gap-2">
          <ProgressBar pct={pct ?? 0} indeterminate={pct == null} className="h-1" />
          {pct != null ? (
            <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink3">{pct}%</span>
          ) : null}
        </div>
      ) : null}

    </button>
  );
});
