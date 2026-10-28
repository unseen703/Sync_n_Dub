import { useCallback, useEffect, useState } from 'react';
import * as Collapsible from '@radix-ui/react-collapsible';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconChevron } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import { fmtRelative } from '../../lib/format';
import { cn } from '../../lib/cn';
import type { EditRecord } from '../../api/types';

function opLabel(op: string): string {
  switch (op) {
    case 'set_voice': return 'Voice';
    case 'set_tone': return 'Delivery';
    case 'set_gender': return 'Gender';
    case 'set_text': return 'Text';
    case 'set_speed': return 'Speed';
    case 'replace_term': return 'Replace';
    case 'retranslate': return 'Re-translate';
    case 'code_mix': return 'Code-mix';
    case 'mute_cue': return 'Mute';
    case 'set_lipsync':
    case 'lipsync': return 'Lip-sync';
    default: return op;
  }
}

function opColor(op: string): string {
  switch (op) {
    case 'set_voice':
    case 'set_gender': return 'bg-violet-500/15 text-violet-700';
    case 'set_tone': return 'bg-blue-500/15 text-blue-700';
    case 'set_text':
    case 'replace_term':
    case 'retranslate':
    case 'code_mix': return 'bg-emerald-500/15 text-emerald-700';
    case 'set_speed': return 'bg-amber-500/15 text-amber-700';
    case 'mute_cue': return 'bg-red-500/15 text-red-700';
    default: return 'bg-ink/10 text-ink3';
  }
}

function RecordCard({ record, admin }: { record: EditRecord; admin: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <div className="rounded-lg border border-line bg-surface/60 px-3 py-2.5">
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <div className="text-[12.5px] font-medium text-ink">
              {record.nlq || `Edit #${record.index + 1}`}
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-1.5">
              {record.ops.map((op, i) => (
                <span
                  key={i}
                  className={cn(
                    'inline-block rounded px-1.5 py-px text-[10px] font-medium',
                    opColor(op.op),
                  )}
                >
                  {opLabel(op.op)}
                  {op.target !== '—' ? ` · ${op.target}` : ''}
                </span>
              ))}
            </div>
            <div className="mt-1 flex items-center gap-1.5 text-[11px] text-ink4">
              {record.mtime ? (
                <span>{fmtRelative(record.mtime)}</span>
              ) : null}
              {record.parsed_by ? (
                <>
                  <span aria-hidden="true">·</span>
                  <span>{record.parsed_by === 'llm' ? 'agent' : record.parsed_by}</span>
                </>
              ) : null}
              {record.dirty_stages.length > 0 ? (
                <>
                  <span aria-hidden="true">·</span>
                  <span>re-ran {record.dirty_stages.join(' → ')}</span>
                </>
              ) : null}
            </div>
          </div>
          <Collapsible.Trigger asChild>
            <button
              type="button"
              aria-label={open ? 'Hide details' : 'Show details'}
              className="rounded-lg p-1.5 text-ink3 transition-colors hover:bg-ink/[0.07] hover:text-ink"
            >
              <IconChevron
                width={13}
                height={13}
                className={cn('transition-transform', open && 'rotate-180')}
              />
            </button>
          </Collapsible.Trigger>
        </div>

        <Collapsible.Content className="overflow-hidden">
          <div className="mt-2 space-y-1.5 border-t border-line/50 pt-2">
            {record.ops.map((op, i) => (
              <div key={i} className="rounded-md bg-canvas/50 px-2.5 py-1.5">
                <div className="font-mono text-[11px] text-ink3">
                  {opLabel(op.op)} → {op.target}
                </div>
                {Object.keys(op.params).length > 0 ? (
                  <div className="mt-0.5 text-[11.5px] text-ink2">
                    {Object.entries(op.params).map(([k, v]) => (
                      <span key={k} className="mr-2">
                        <span className="text-ink4">{k}:</span>{' '}
                        {String(v)}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ))}
            {record.warnings.length > 0 ? (
              <div className="space-y-1">
                {record.warnings.map((w, i) => (
                  <p key={i} className="flex items-start gap-1.5 text-[11.5px] text-warn">
                    <span aria-hidden="true">⚠</span>
                    <span>{w}</span>
                  </p>
                ))}
              </div>
            ) : null}
            {admin && Object.keys(record.speaker_voice_overrides).length > 0 ? (
              <div className="rounded-md bg-canvas/50 px-2.5 py-1.5">
                <div className="font-mono text-[11px] text-ink3">Voice overrides</div>
                <pre className="mt-0.5 text-[11px] text-ink2">
                  {JSON.stringify(record.speaker_voice_overrides, null, 2)}
                </pre>
              </div>
            ) : null}
          </div>
        </Collapsible.Content>
      </div>
    </Collapsible.Root>
  );
}

export function HistoryPanel() {
  const history = useStore((s) => s.editHistory);
  const refreshHistory = useStore((s) => s.refreshEditHistory);
  const admin = useStore((s) => s.mode === 'admin');
  const [open, setOpen] = useState(false);

  const load = useCallback(() => {
    void refreshHistory();
  }, [refreshHistory]);

  useEffect(() => { load(); }, [load]);

  const edits = history?.edits ?? [];
  const hasEdits = edits.length > 0;

  return (
    <section className="card p-4">
      <Collapsible.Root open={open} onOpenChange={setOpen}>
        <div className="flex items-center gap-2">
          <Collapsible.Trigger asChild>
            <button
              type="button"
              className="flex items-center gap-1.5 text-ink transition-colors hover:text-ink"
            >
              <IconChevron
                width={13}
                height={13}
                className={cn('text-ink3 transition-transform', open && 'rotate-180')}
              />
              <Eyebrow>
                Edit history{hasEdits ? ` · ${edits.length}` : ''}
              </Eyebrow>
            </button>
          </Collapsible.Trigger>

          {history?.outdated ? (
            <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10.5px] font-medium text-amber-700">
              Output outdated
            </span>
          ) : null}

          {history?.is_running ? (
            <span className="rounded-full bg-brand/15 px-2 py-0.5 text-[10.5px] font-medium text-brand-ink">
              Re-rendering
            </span>
          ) : null}

          <Button size="sm" variant="ghost" className="ml-auto" onClick={load}>
            Refresh
          </Button>
        </div>

        <Collapsible.Content className="overflow-hidden">
          <div className="mt-3 space-y-2">
            {!hasEdits ? (
              <p className="text-[11.5px] text-ink4">
                No edits have been applied yet. Changes you apply will appear here
                with what was changed and which stages re-ran.
              </p>
            ) : (
              <>
                {history?.outdated ? (
                  <p className="mb-2 text-[11.5px] text-amber-600">
                    Pending edits exist that are not yet reflected in the current output.
                  </p>
                ) : null}
                {[...edits].reverse().map((r) => (
                  <RecordCard key={r.file} record={r} admin={admin} />
                ))}
              </>
            )}

            {admin && (history?.snapshots?.length ?? 0) > 0 ? (
              <div className="mt-3 border-t border-line/50 pt-3">
                <Eyebrow className="mb-2">Transcript versions</Eyebrow>
                <div className="space-y-1">
                  {history!.snapshots.map((s) => (
                    <div
                      key={s.file}
                      className="flex items-center gap-2 rounded-md bg-canvas/50 px-2.5 py-1.5 text-[11.5px]"
                    >
                      <span className="font-mono text-ink3">{s.producer}</span>
                      <span className="text-ink4">·</span>
                      <span className="text-ink4">{fmtRelative(s.mtime)}</span>
                      <span className="ml-auto truncate font-mono text-[10px] text-ink4">
                        {s.file}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </Collapsible.Content>
      </Collapsible.Root>
    </section>
  );
}
