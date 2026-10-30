import { useEffect, useState } from 'react';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { Eyebrow, Skeleton } from '../ui/Card';
import { IconAlert, IconBolt } from '../ui/Icons';
import { useStore, toStagedDiffs } from '../../store/useStore';
import { toneLabel } from '../../lib/format';
import { fmtDuration } from '../../lib/format';
import type { EditPlan } from '../../api/types';

export function ApplyDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const staged = useStore((s) => s.staged);
  const plan = useStore((s) => s.editPlan);
  const previewPlan = useStore((s) => s.previewPlan);
  const applyStaged = useStore((s) => s.applyStaged);
  const applying = useStore((s) => s.applying);
  const speakers = useStore((s) => s.speakers);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    void previewPlan().finally(() => setLoading(false));
  }, [open, previewPlan]);

  const diffs = toStagedDiffs(staged);

  const nameFor = (id: string) =>
    speakers?.speakers.find((s) => s.id === id)?.label ?? id;

  const toneFor = (v: string | null) => toneLabel(speakers?.tone_options, v);
  const nameOf = (voice: string | null) =>
    voice
      ? (speakers?.voice_options.find((o) => o.voice === voice)?.display ?? voice.split('-').pop())
      : '—';

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      width="max-w-xl"
      title="Apply these changes and re-render?"
      description={
         'Review the changes made and what will run again.'
      }
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            busy={applying}
            onClick={async () => {
              await applyStaged();
              onOpenChange(false);
            }}
          >
            Apply &amp; re-render
            {plan?.cost?.total_est_s ? ` · ~${fmtDuration(plan.cost.total_est_s)}` : ''}
          </Button>
        </>
      }
    >
      <div className="space-y-5">
        <section>
          <Eyebrow className="mb-2">What you changed</Eyebrow>
          <ul className="space-y-1.5">
            {diffs.map((e) => (
              <li key={`${e.speaker}-${e.field}`} className="flex items-center gap-2 text-[12.5px]">
                <span className="min-w-0 truncate font-medium text-ink">
                  {nameFor(e.speaker)}
                </span>
                <span className="text-ink4">{e.field === 'voice' ? 'voice' : 'delivery'}</span>
                <span className="text-ink3">
                  {e.field === 'voice' ? nameOf(e.from) : toneFor(e.from)}
                </span>
                <span className="text-ink4" aria-hidden="true">→</span>
                <span className="font-semibold text-ink">
                  {e.field === 'voice' ? nameOf(e.to) : toneFor(e.to)}
                </span>
              </li>
            ))}
          </ul>
        </section>

        {loading && !plan ? (
          <Skeleton className="h-28" />
        ) : plan ? (
          <>
            <CostSection plan={plan} />

            {(plan.warnings ?? []).length ? (
              <section className="rounded-xl border border-warn/40 bg-warn/[0.07] px-3 py-2.5">
                {plan.warnings.map((w) => (
                  <p key={w} className="flex gap-2 text-[12.5px] leading-relaxed text-ink2">
                    <IconAlert width={13} height={13} className="mt-px shrink-0 text-warn" />
                    {w}
                  </p>
                ))}
              </section>
            ) : null}
          </>
        ) : (
          <p className="t-sm">
            We could not work out what this change costs. Applying it is still safe — the
            pipeline re-runs only the stages the change touches — but you will not see an
            estimate first.
          </p>
        )}
      </div>
    </Dialog>
  );
}

function CostSection({ plan }: { plan: EditPlan }) {

  const stages = plan.cost?.stages ?? [];
  if (!stages.length) {
    return (
      <section>
        <Eyebrow className="mb-2">What re-runs</Eyebrow>
        <p className="t-sm">
          {plan.dirty_stages.length
            ? plan.dirty_stages.join(' → ')
            : 'Nothing — these changes need no re-render.'}
        </p>
      </section>
    );
  }
  return (
    <section>
      <Eyebrow className="mb-2">What re-runs</Eyebrow>
      <ul className="divide-y divide-line overflow-hidden rounded-xl border border-line">
        {stages.map((s) => (
          <li key={s.id} className="flex items-center gap-2 px-3 py-2 text-[12.5px]">
            <span className="min-w-0 flex-1 truncate text-ink2">{s.label}</span>
            {s.gpu ? (
              <span
                title="Runs on the GPU — minutes, not seconds."
                className="inline-flex items-center gap-1 rounded border border-aqua/35 bg-aqua/10 px-1.5 py-px text-[11px] text-aqua-ink"
              >
                <IconBolt width={9} height={9} />
                uses the GPU
              </span>
            ) : null}
            <span className="w-20 shrink-0 text-right font-mono text-[11.5px] tabular-nums text-ink3">
              {s.est_s ? fmtDuration(s.est_s) : '—'}
            </span>
          </li>
        ))}
      </ul>
      {plan.cost?.total_est_s ? (
        <p className="mt-2 text-[12.5px] text-ink3">
          Total: <span className="font-semibold text-ink">
            about {fmtDuration(plan.cost.total_est_s)}
          </span>
        </p>
      ) : null}
    </section>
  );
}
