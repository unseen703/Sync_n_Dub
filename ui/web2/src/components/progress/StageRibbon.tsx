import { m } from 'framer-motion';
import { useStore } from '../../store/useStore';
import { cn } from '../../lib/cn';
import { fmtDuration } from '../../lib/format';
import { IconAlert, IconCheck } from '../ui/Icons';
import { Skeleton } from '../ui/Card';
import type { Phase } from '../../api/types';

/**
 * The pipeline as a numbered step bar.
 *
 *   (✓)━━━━(✓)━━━━(3)────(4)────(5)────(6)────(7)
 *  Download Speaker Transcribe Translate Speech Mix  Lip sync
 *
 * Seven nodes on one line, the connector between two nodes painted once the
 * stage behind it finishes. A connector is owned by the stage to its LEFT,
 * so the line and the circles can never disagree about how far along the run
 * is — there is one source of truth (`phase.state`) and two renderings of it.
 *
 * `ui/progress.py` folds the pipeline's 24 internal markers into these seven,
 * so a clip-runner job (STATUS.json, different phase names) and a pipeline
 * job read identically here. Deliberately not a tree: the sub-stages are real
 * and still travel in the payload, but `9f1b: viseme extraction (mms_char)`
 * answers a question nobody watching a dub is asking.
 *
 * The current node pulses instead of showing a percentage, because there is
 * no honest percentage to show: stage 9h is most of the wall time and shells
 * into another conda env, so its internal progress reaches no logger of ours.
 */

function isDone(p: Phase): boolean {
  return p.state === 'done' || p.state === 'skipped';
}

/** A connector is painted when the stage on its left has finished. */
function Connector({ side, lit, failed }: {
  side: 'left' | 'right';
  lit: boolean;
  failed: boolean;
}) {
  return (
    <div
      className={cn(
        'absolute top-[12px] h-[2px] -translate-y-1/2',
        side === 'left' ? 'left-0 right-1/2' : 'left-1/2 right-0',
        'bg-ink/[0.12]',
      )}
    >
      <m.div
        className={cn(
          'h-full rounded-full',
          failed ? 'bg-danger-fill' : 'bg-gradient-to-r from-brand to-aqua',
        )}
        initial={false}
        animate={{ width: lit ? '100%' : '0%' }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      />
    </div>
  );
}

export function StageRibbon() {
  const progress = useStore((s) => s.progress);

  if (!progress) {
    return (
      <section className="card px-5 py-4">
        <Skeleton className="h-14 border-0 bg-ink/[0.05]" />
      </section>
    );
  }

  const phases = progress.phases ?? [];
  if (!phases.length) {
    return (
      <section className="card px-5 py-3.5">
        <p className="t-sm">
          {progress.state === 'done'
            ? 'Finished. This job predates stage tracking, so there is no breakdown.'
            : 'This dub is still starting up — the stage breakdown appears once the first stage logs.'}
        </p>
      </section>
    );
  }

  const n = phases.length;
  const doneCount = phases.filter(isDone).length;
  const runningIdx = phases.findIndex((p) => p.state === 'running');
  const failedIdx = phases.findIndex((p) => p.state === 'failed');
  const allDone = progress.state === 'done' || doneCount === n;
  const current = failedIdx >= 0 ? phases[failedIdx] : (phases[runningIdx] ?? null);
  const elapsed = phases.reduce((a, p) => a + (p.wall_s ?? 0), 0);

  return (
    <section className="card px-5 pb-4 pt-4" aria-label="Pipeline progress">
      <div className="mb-3 flex items-baseline gap-2">
        <span
          className={cn(
            'text-[14px] font-semibold',
            failedIdx >= 0 ? 'text-danger' : 'text-ink',
          )}
        >
          {failedIdx >= 0
            ? `Failed · ${current?.label}`
            : allDone
              ? 'Finished'
              : (current?.label ?? 'Starting')}
        </span>
        {failedIdx < 0 ? (
          <span className="text-[12px] text-ink3">
            {allDone ? `all ${n} stages` : `${doneCount} of ${n} stages`}
          </span>
        ) : null}
        {elapsed > 0 ? (
          <span className="ml-auto font-mono text-[12px] tabular-nums text-ink3">
            {fmtDuration(elapsed)}
          </span>
        ) : null}
      </div>

      <div
        className="flex"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={n}
        aria-valuenow={doneCount}
        aria-label={`Pipeline progress: ${doneCount} of ${n} stages complete`}
      >
        {phases.map((p, i) => {
          const done = isDone(p);
          const running = i === runningIdx && failedIdx < 0;
          const failed = i === failedIdx;
          const prevDone = i > 0 && isDone(phases[i - 1]);
          return (
            <div
              key={p.id}
              className="relative flex min-w-0 flex-1 flex-col items-center"
              title={p.wall_s ? `${p.label} · ${fmtDuration(p.wall_s)}` : p.label}
            >
              {i > 0 ? (
                <Connector side="left" lit={prevDone} failed={failed} />
              ) : null}
              {i < n - 1 ? (
                <Connector side="right" lit={done} failed={false} />
              ) : null}

              {/* the node */}
              <div className="relative z-10">
                {running ? (
                  <m.span
                    className="absolute inset-0 rounded-full bg-aqua/45"
                    animate={{ opacity: [0.7, 0, 0.7], scale: [1, 1.8, 1] }}
                    transition={{ duration: 1.7, repeat: Infinity, ease: 'easeInOut' }}
                  />
                ) : null}
                <span
                  className={cn(
                    'relative flex h-[24px] w-[24px] items-center justify-center',
                    'rounded-full text-[11.5px] font-semibold tabular-nums',
                    failed
                      ? 'bg-danger-fill text-white'
                      : done
                        ? 'bg-gradient-to-br from-brand to-aqua text-white'
                        : running
                          ? 'border-2 border-aqua bg-canvas text-aqua-ink'
                          : 'border border-edge/70 bg-canvas text-ink4',
                  )}
                >
                  {failed ? (
                    <IconAlert width={11} height={11} />
                  ) : done ? (
                    <IconCheck width={11} height={11} />
                  ) : (
                    i + 1
                  )}
                </span>
              </div>

              <span
                className={cn(
                  'mt-2 block w-full truncate px-0.5 text-center',
                  'text-[11.5px] leading-tight',
                  failed
                    ? 'font-semibold text-danger'
                    : running
                      ? 'font-semibold text-ink'
                      : done
                        ? 'text-ink3'
                        : 'text-ink4',
                )}
              >
                {p.label}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
