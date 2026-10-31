import { useState } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { useStore } from '../../store/useStore';
import { cn } from '../../lib/cn';
import { fmtDuration } from '../../lib/format';
import { IconAlert, IconCheck, IconChevron } from '../frontend/Icons';
import { Skeleton } from '../frontend/Card';
import type { Phase, Stage } from '../../api/types';

function isComplete(p: Phase | Stage): boolean {
  return p.state === 'done' || p.state === 'skipped';
}

function isDone(p: Phase | Stage): boolean {
  return p.state === 'done';
}

function isDeferred(p: Phase): boolean {
  return p.state === 'pending' && Boolean(p.deferred);
}

function hasDetail(p: Phase): boolean {
  return p.substages.length > 0 || (p.contains?.length ?? 0) > 0;
}

function Connector({ side, lit, failed }: {
  side: 'left' | 'right';
  lit: boolean;
  failed: boolean;
}) {
  return (
    <div
      className={cn(
        'absolute top-[12px] h-[2px] -translate-y-1/2',
        side === 'left'
          ? 'left-0 right-[calc(50%+12px)]'
          : 'left-[calc(50%+12px)] right-0',
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

function stateIcon(stage: Stage) {
  const { state } = stage;
  if (state === 'done') return <IconCheck width={10} height={10} className="text-brand" />;
  if (state === 'failed') return <IconAlert width={10} height={10} className="text-danger" />;
  if (state === 'running') return <span className="inline-block h-1.5 w-1.5 rounded-full bg-aqua animate-pulse" />;
  if (state === 'skipped') return <span className="text-[10px] text-ink4">skip</span>;

  if (state === 'pending' && stage.deferred) return <span className="text-[10px] text-ink4">later</span>;
  return null;
}

function SubstageRow({ stage }: { stage: Stage }) {
  return (
    <div className="flex items-center gap-2 py-0.5">
      <span className="flex w-4 shrink-0 items-center justify-center">
        {stateIcon(stage)}
      </span>
      <span className={cn(
        'text-[12px] leading-tight',
        stage.state === 'done' ? 'text-ink3' : stage.state === 'running' ? 'text-ink font-medium' : 'text-ink4',
      )}>
        {stage.label}
      </span>
      {stage.wall_s ? (
        <span className="ml-auto font-mono text-[11px] tabular-nums text-ink4">
          {fmtDuration(stage.wall_s)}
        </span>
      ) : null}
    </div>
  );
}

/** Pipeline progress ribbon. It reads job phases and renders their current states and details. */
export function StageRibbon() {
  const progress = useStore((s) => s.progress);
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!progress) {
    return (
      <section className="card px-5 py-4">
        <Skeleton className="h-14 border-0 bg-ink/[0.05]" />
      </section>
    );
  }

  const atGate = Boolean(progress.milestones?.awaiting_cast);

  // Every UI job runs the full pipeline by default -- there is no "this run
  // intentionally skipped lip-sync" case here, so every `skipped` pill really
  // means "not run yet". The backend's `_group_phases` writes `skipped` as
  // the default for an empty group and `_finished_phase` counts it as
  // complete, which rolls Finished up to `done` and lights the connector
  // line through pills that have not run. Normalise `skipped` → `pending`
  // across the whole ribbon so the state is honest end to end.
  const rawPhases = progress.phases ?? [];
  const anySkipped = rawPhases.some((p) => p.id !== 'finished' && p.state === 'skipped');
  const phases = rawPhases.map((p) => {
    if (p.id !== 'finished' && p.state === 'skipped') {
      return { ...p, state: 'pending' as const };
    }
    // Finished → pending whenever any upstream pill was treated as skipped
    // (now pending), or we are still at the cast gate.
    if (p.id === 'finished' && p.state === 'done' && (atGate || anySkipped)) {
      return { ...p, state: 'pending' as const };
    }
    return p;
  });
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
  const liveCount = phases.filter((p) => p.state !== 'skipped').length;

  const lastDoneIdx = phases.reduce((acc, p, i) => (isDone(p) ? i : acc), -1);
  const runningIdx = phases.findIndex((p) => p.state === 'running');
  const failedIdx = phases.findIndex((p) => p.state === 'failed');
  const allDone = n > 0 && phases.every(isComplete);
  const current = failedIdx >= 0 ? phases[failedIdx] : (phases[runningIdx] ?? null);
  const pending = !current && !allDone
    ? phases.find((p) => p.state === 'pending' && p.id !== 'finished') ?? null
    : null;
  const elapsed = phases.reduce((a, p) => a + (p.wall_s ?? 0), 0);

  const toggle = (id: string) => {
    setExpanded((prev) => prev === id ? null : id);
  };

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
            : atGate
              // Checked first: a job awaiting cast is never "Finished", even
              // if a planning gap once let the Dub-audio/Lip-sync groups come
              // back empty-and-therefore-`skipped` (see `plan_stages` in
              // `frontend/progress.py`), which made `allDone` true for a dub that
              // had not been cast yet.
              ? 'Translation complete. Please select the voices as per your requirement to proceed with the dubbed audio and video generation.'
              : allDone
                ? 'Finished'
                : pending
                  ? `${pending.label} not started`
                  : (current?.label ?? 'Starting')}
        </span>
        {failedIdx < 0 ? (
          <span className="text-[12px] text-ink3">
            {allDone ? `all ${liveCount} stages` : `${doneCount} of ${liveCount} stages`}
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
        aria-valuemax={liveCount}
        aria-valuenow={doneCount}
        aria-label={`Pipeline progress: ${doneCount} of ${liveCount} stages complete`}
      >
        {phases.map((p, i) => {
          const done = isDone(p);
          const skipped = p.state === 'skipped';
          const deferred = isDeferred(p);
          const running = i === runningIdx && failedIdx < 0;
          const failed = i === failedIdx;

          const lineEnd = i <= lastDoneIdx;
          const prevDone = i > 0 && i - 1 <= lastDoneIdx;
          const expandable = hasDetail(p);
          const isExpanded = expanded === p.id;
          return (
            <div
              key={p.id}
              className="relative flex min-w-0 flex-1 flex-col items-center"
              title={
                deferred
                  ? `${p.label} — not run yet`
                  : skipped
                    ? `${p.label} — not part of this run`
                    : p.wall_s
                      ? `${p.label} · ${fmtDuration(p.wall_s)}`
                      : p.label
              }
            >
              {i > 0 ? (
                <Connector side="left" lit={prevDone} failed={failed} />
              ) : null}
              {i < n - 1 ? (
                <Connector side="right" lit={lineEnd} failed={false} />
              ) : null}

              <button
                type="button"
                className={cn('relative z-10', expandable ? 'cursor-pointer' : 'cursor-default')}
                onClick={() => expandable && toggle(p.id)}
                aria-expanded={expandable ? isExpanded : undefined}
                tabIndex={expandable ? 0 : -1}
              >
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

                          : skipped
                            ? 'border border-edge/50 bg-ink/[0.03] text-ink4/70'
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
              </button>

              <span
                className={cn(
                  'mt-2 flex w-full items-center justify-center gap-0.5 truncate px-0.5 text-center',
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
                {expandable ? (
                  <IconChevron
                    width={10}
                    height={10}
                    className={cn(
                      'shrink-0 transition-transform duration-200',
                      isExpanded && 'rotate-180',
                    )}
                  />
                ) : null}
              </span>
            </div>
          );
        })}
      </div>

      <AnimatePresence initial={false}>
        {expanded && (() => {
          const phase = phases.find((p) => p.id === expanded);
          if (!phase || !hasDetail(phase)) return null;
          return (
            <m.div
              key={expanded}
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              className="overflow-hidden"
            >
              <div className="mt-3 rounded-lg border border-edge/50 bg-ink/[0.02] px-3 py-2">
                <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-ink4">
                  {phase.label} — detail
                </p>
                {phase.substages.map((s) => (
                  <SubstageRow key={s.key} stage={s} />
                ))}

                {(phase.contains ?? []).map((c) => (
                  <div key={c.key} className="flex items-center gap-2 py-0.5">
                    <span className="flex w-4 shrink-0 items-center justify-center">
                      <span className="text-[10px] text-ink4/70" aria-hidden>–</span>
                    </span>
                    <span className="text-[12px] leading-tight text-ink4/70">
                      {c.label}
                    </span>
                    <span className="ml-auto text-[11px] text-ink4/70">
                      not in this run
                    </span>
                  </div>
                ))}
              </div>
            </m.div>
          );
        })()}
      </AnimatePresence>
    </section>
  );
}
