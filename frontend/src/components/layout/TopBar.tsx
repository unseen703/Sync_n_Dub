import { useState } from 'react';
import { useStore } from '../../store/useStore';
import { Button } from '../frontend/Button';
import { StateBadge } from '../frontend/StateBadge';
import { IconStop } from '../frontend/Icons';
import { fmtElapsed, jobTitle, normalizeState } from '../../lib/format';
import { useNow } from '../../hooks/useNow';
import { cancelJob, MOCKED } from '../../api';
import { cn } from '../../lib/cn';

/** Current-job header. It reads job state and renders its title, status, and available actions. */
export function TopBar() {
  const selected = useStore((s) => s.selected);
  const job = useStore((s) => s.jobs.find((j) => j.job_id === s.selected) ?? null);

  const spec = useStore((s) => s.spec);
  const progress = useStore((s) => s.progress);
  const refreshProgress = useStore((s) => s.refreshProgress);
  const pushToast = useStore((s) => s.pushToast);
  const [cancelling, setCancelling] = useState(false);

  const running = normalizeState(progress?.state) === 'running';
  const now = useNow(running);

  const title = spec?.title ?? (job ? jobTitle(job).text : null);
  const degraded = !spec?.title && job ? jobTitle(job).degraded : false;

  return (
    <div className="flex items-center gap-3 border-b border-line px-3 py-2.5 lg:px-5">
      <div className="min-w-0 flex-1">
        {title ? (
          <h1
            className={cn(
              'truncate text-[15px] font-semibold leading-tight',
              degraded ? 'font-mono text-[13px] text-ink3' : 'text-ink',
            )}
            title={title}
          >
            {title}
          </h1>
        ) : (
          <div className="h-3.5 w-48 rounded bg-ink/[0.08]" />
        )}
        {selected ? (
          <div className="truncate text-[12.5px] text-ink4" title={selected}>
            {selected}
          </div>
        ) : null}
      </div>

      {progress ? (
        <div className="flex shrink-0 items-center gap-2.5">
          {running ? (
            <span className="hidden font-mono text-[13px] tabular-nums text-ink3 sm:inline">
              {progress.pct}% · {fmtElapsed(progress.started_at, now)}
            </span>
          ) : null}
          <StateBadge state={progress.state} />

          {running ? (
            <Button
              size="sm"
              variant="danger"
              busy={cancelling}
              title={
                MOCKED.cancel
                  ? 'POST /api/jobs/{id}/cancel is not built yet — this will not stop the run.'
                  : 'Stop this run. Everything finished so far is kept.'
              }
              onClick={async () => {
                if (!selected) return;
                setCancelling(true);
                try {
                  await cancelJob(selected);
                  await refreshProgress();
                  pushToast({
                    kind: 'warn',
                    title: 'Stopping this dub',
                    body: 'Everything the run already finished is kept on disk.',
                  });
                } catch (e) {
                  pushToast({
                    kind: 'error',
                    title: 'Could not stop the run',
                    body: (e as Error).message,
                  });
                } finally {
                  setCancelling(false);
                }
              }}
            >
              <IconStop width={11} height={11} />
              Stop
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
