import { useCallback, useEffect, useState } from 'react';
import { Button } from '../frontend/Button';
import { Eyebrow } from '../frontend/Card';
import { Dialog } from '../frontend/Dialog';
import { SpeakerTable } from '../speakers/SpeakerTable';
import { VideoStage } from '../video/VideoStage';
import { NowPlayingCues } from '../video/NowPlayingCues';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import { fmtClock } from '../../lib/format';
import { ActionWorkspace } from './ActionWorkspace';
import { LipsyncPanel } from './LipsyncPanel';
import { HistoryPanel } from './HistoryPanel';
import type {
  AgentScript,
  AgentScriptRow,
  AgentSpeaker,
  AgentTurn,
  CostPreview,
  ScriptEdit,
} from '../../api/types';

function scoped(
  text: string,
  selectedCue: string | null,
  selectedSpeaker: string | null,
  script: AgentScript | null,
): string {
  if (!text) return text;
  const alreadyScoped =
    /\bcue\s*#?\d+|#\d+\b|\bseg(?:ment)?[_ ]?\d+|\b(?:SPK|SPEAKER)[_ ]?\d+\b|\d{1,2}:\d{2}/i.test(
      text,
    );
  if (alreadyScoped) return text;
  if (selectedCue) {
    const row = (script?.segments ?? []).find((r) => r.id === selectedCue);
    if (row) return `cue ${row.cue_no}: ${text}`;
  }
  if (selectedSpeaker) return `${selectedSpeaker}: ${text}`;
  return text;
}

/** Editing workspace. Input: job id. Output: cast, script, staged edits, and post-render tools for that job. */
export function EditPanel({ jobId }: { jobId: string }) {
  const pushToast = useStore((s) => s.pushToast);
  const params = useStore((s) => s.agentParams);
  const refreshParams = useStore((s) => s.refreshAgentParams);
  const refreshSpeakers = useStore((s) => s.refreshSpeakers);
  const refreshHistory = useStore((s) => s.refreshEditHistory);
  const selectedCue = useStore((s) => s.selectedCue);
  const selectedSpeaker = useStore((s) => s.selectedSpeaker);
  const [script, setScript] = useState<AgentScript | null>(null);
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [, s] = await Promise.all([
      refreshParams(),
      api.getAgentScript(jobId),
    ]);
    setScript(s);
  }, [jobId, refreshParams]);

  const dropTurn = useCallback(async (index: number) => {
    setBusy(true);
    setError(null);
    try {
      await api.dropStagedTurn(jobId, index);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [jobId, load]);

  const muteCue = useCallback(async (segId: string, on: boolean) => {
    setBusy(true);
    setError(null);
    try {
      await api.muteCue(jobId, segId, on);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [jobId, load]);

  const stageScript = useCallback(async (edits: ScriptEdit[]) => {
    if (!edits.length) return;
    setBusy(true);
    setError(null);
    try {
      const plan = await api.stageScript(jobId, edits);
      pushToast({
        kind: 'success',
        title: `${edits.length} line${edits.length === 1 ? '' : 's'} staged`,
        body: `Applying replays ${plan.dirty_stages.join(' → ')}.`,
      });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [jobId, load, pushToast]);

  useEffect(() => {
    void load().catch((e) => setError((e as Error).message));
  }, [load]);

  const send = async () => {
    const text = scoped(msg.trim(), selectedCue, selectedSpeaker, script);
    if (!text) return;
    setBusy(true);
    setError(null);
    try {
      await api.agentSend(jobId, text);
      setMsg('');
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const apply = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.applyEdits(jobId);
      const stages = r.dirty_stages ?? [];
      pushToast({
        kind: 'success',
        title: 'Applied',
        body: stages.length
          ? `Re-running: ${stages.join(' → ')}`
          : 'Edit committed — no re-render needed. Use "Start TTS & mix" when ready.',
      });
      await load();

      void refreshSpeakers();
      void refreshHistory();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const discard = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.discardEdits(jobId);
      pushToast({ kind: 'success', title: 'Discarded', body: 'All pending edits removed.' });
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // Apply is clickable whenever the agent has pending ops. Edit-only ops
  // (create_speaker) have `dirty_stages == []` — they commit the edit but
  // trigger no re-render; the button must still appear.
  const hasPendingOps = (params?.turns?.length ?? 0) > 0;
  const staged = hasPendingOps;

  const pendingMutes = new Set<string>();
  for (const turn of params?.turns ?? []) {
    for (const op of turn.ops ?? []) {
      if ((op as { op?: string }).op !== 'mute_cue') continue;
      const segs = ((op as { scope?: { segments?: unknown[] } }).scope?.segments ?? []) as unknown[];
      for (const sid of segs) {
        if (typeof sid === 'string') pendingMutes.add(sid);
      }
    }
  }

  return (
    <div className="space-y-4">

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">

        <div className="min-w-0 space-y-4">
          <VideoStage />

          <ActionWorkspace
            msg={msg}
            setMsg={setMsg}
            onSend={() => void send()}
            busy={busy}
            error={error}
            warnings={params?.warnings ?? []}
            actions={params?.actions ?? []}
            script={script?.segments ?? []}
            voiceOptions={params?.voice_options ?? []}
            toneOptions={params?.tone_options ?? []}
            speakers={params?.speakers ?? []}
          />

          <StagedChanges
            turns={params?.turns ?? []}
            script={script}
            speakers={params?.speakers ?? []}
            busy={busy}
            onDrop={(i) => void dropTurn(i)}
            canApply={staged}
            dirtyStages={params?.dirty_stages ?? []}
            cost={params?.cost ?? null}
            onApply={() => void apply()}
            onDiscard={() => void discard()}
          />
        </div>

        <div className="min-w-0 space-y-4">
          <NowPlayingCues
            segments={script?.segments ?? []}
            busy={busy}
            mutedIds={pendingMutes}
            onMute={(segId, on) => void muteCue(segId, on)}
            onStageScript={(edits) => void stageScript(edits)}
          />
          <SpeakerTable scroll segments={script?.segments ?? []} />
        </div>
      </div>

      <LipsyncPanel jobId={jobId} onStarted={() => void load()} />

      <HistoryPanel jobId={jobId} />
    </div>
  );
}

interface Diff {
  object: string;
  before: string;
  after: string;

  pending?: boolean;
}

function opDiffs(
  op: Record<string, unknown>,
  segById: Map<string, AgentScriptRow>,
  speakers: AgentSpeaker[],
): Diff[] {
  const name = String(op?.op ?? '').toLowerCase();
  const scope = (op?.scope ?? {}) as Record<string, unknown>;
  const params = (op?.params ?? {}) as Record<string, unknown>;

  const segsRaw = scope.segments ?? scope.segment_ids;
  const segIds = Array.isArray(segsRaw)
    ? segsRaw.map(String)
    : segsRaw
      ? [String(segsRaw)]
      : [];
  const rows = segIds.map((id) => segById.get(id)).filter(Boolean) as AgentScriptRow[];

  let tgt: string;
  if (rows.length) {
    tgt = rows.length <= 2
      ? rows.map((r) => `cue #${r.cue_no}`).join(', ')
      : `${rows.length} cues (from #${rows[0].cue_no})`;
  } else if (scope.speaker) {
    tgt = String(scope.speaker);
  } else if (scope.all) {
    tgt = 'whole dub';
  } else {
    const s = scope.start_s ?? scope.start;
    const e = scope.end_s ?? scope.end;
    if (s != null && e != null) tgt = `${fmtClock(Number(s))}–${fmtClock(Number(e))}`;
    else if (s != null) tgt = `at ${fmtClock(Number(s))}`;
    else tgt = 'the dub';
  }

  const spk = scope.speaker
    ? speakers.find((s) => s.id === String(scope.speaker)) ?? null
    : null;
  const spkName = spk?.label || (scope.speaker ? String(scope.speaker) : tgt);
  const firstText = rows[0]?.text ?? '';

  switch (name) {
    case 'set_text':
      return [{ object: tgt, before: firstText, after: String(params.text ?? '') }];
    case 'replace_term': {
      const find = String(params.find ?? params.term ?? params.search ?? '');
      const repl = String(params.replace ?? params.with ?? '');
      return [{
        object: tgt,
        before: find ? `"${find}"` : firstText,
        after: repl ? `"${repl}"` : '(removed)',
      }];
    }
    case 'set_speed': {
      const factor = Number(params.factor ?? params.speed);
      const before = rows[0]?.speed != null ? `${rows[0].speed.toFixed(2)}×` : '1.00×';
      return [{
        object: `${tgt} · speed`,
        before,
        after: Number.isFinite(factor) ? `${factor.toFixed(2)}×` : '—',
      }];
    }
    case 'set_voice':
      return [{
        object: `${spkName} · voice`,
        before: spk?.voice.current ?? '—',
        after: String(params.voice ?? spk?.voice.staged ?? '—'),
      }];
    case 'set_gender':
      return [{
        object: `${spkName} · gender`,
        before: spk?.gender.current ?? '—',
        after: String(params.gender ?? spk?.gender.staged ?? '—'),
      }];
    case 'set_tone':
      return [{
        object: `${spkName} · delivery`,
        before: spk?.tone.current ?? '—',
        after: String(params.tone ?? spk?.tone.staged ?? '—'),
        pending: Boolean(params.rewrite_text),
      }];
    case 'retranslate': {
      const kind = String((op as Record<string, unknown>).kind ?? '');
      const tgtLang = String(params.target_language ?? '').trim();
      const srcText = String(rows[0]?.src_text ?? '').trim();
      // Script path: target language matches the dub's source language, so
      // the handler copies the ASR line verbatim into mt_text — no LLM
      // round trip. Show the actual before→after if we have the ASR text.
      if (kind === 'script') {
        if (!srcText) {
          return [{
            object: tgt,
            before: firstText,
            after: 'source cue is empty — no change applied',
          }];
        }
        return [{
          object: tgt + (tgtLang ? ` · ${tgtLang}` : ''),
          before: firstText,
          after: srcText,
        }];
      }
      return [{
        object: tgt + (tgtLang ? ` · ${tgtLang}` : ''),
        before: firstText,
        after: 're-translated on apply',
        pending: true,
      }];
    }
    case 'code_mix':
      return [{
        object: tgt,
        before: firstText,
        after: `code-mixed${params.mode ? ` (${String(params.mode)})` : ''}`,
        pending: true,
      }];
    case 'mute_cue':
      return [{ object: tgt, before: firstText || 'this cue', after: 'muted', pending: true }];
    case 'set_lipsync':
    case 'lipsync':
      return [{
        object: 'Lip-sync model',
        before: '—',
        after: String(params.model ?? params.lipsync ?? '—'),
      }];
    default:
      return [{ object: tgt, before: '', after: name || 'change' }];
  }
}

function sourceLabel(source: string): string {
  switch (source) {
    case 'script': return 'direct edit';
    case 'llm': return 'parsed by LLM';
    case 'rules': return 'parsed by rules';
    default: return source;
  }
}

function kindChipClass(kind: string): string {
  switch (kind) {
    case 'rule':
      return 'bg-sky-500/15 text-sky-700';
    case 'script':
      return 'bg-amber-500/15 text-amber-700';
    case 'llm':
      return 'bg-fuchsia-500/15 text-fuchsia-700';
    default:
      return 'bg-ink1/10 text-ink3';
  }
}

function kindTitle(kind: string): string {
  switch (kind) {
    case 'rule':
      return 'rule: handler mutates deterministically, no outside call';
    case 'script':
      return 'script: the replay re-runs a pipeline stage via an external model';
    case 'llm':
      return 'llm: the handler rewrites text through the LLM at apply time';
    default:
      return kind;
  }
}

function StagedChanges({
  turns,
  script,
  speakers,
  busy,
  onDrop,
  canApply,
  dirtyStages,
  cost,
  onApply,
  onDiscard,
}: {
  turns: AgentTurn[];
  script: AgentScript | null;
  speakers: AgentSpeaker[];
  busy: boolean;
  onDrop: (index: number) => void;
  canApply: boolean;
  dirtyStages: string[];
  cost: CostPreview | null;
  onApply: () => void;
  onDiscard: () => void;
}) {

  const [confirming, setConfirming] = useState<number | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);

  const segById = new Map((script?.segments ?? []).map((r) => [r.id, r]));

  return (
    <section className="card p-4">
      <Eyebrow className="mb-2">
        Pending edits{turns.length ? ` · ${turns.length}` : ''}
      </Eyebrow>
      {turns.length > 0 && (
        <p className="mb-2 text-[11px] text-amber-600">
          Not yet included in the rendered video.
        </p>
      )}

      {turns.length === 0 ? (
        <p className="text-[11.5px] text-ink4">
          Changes appear here. Nothing re-renders until you apply.
        </p>
      ) : (
        <div className="max-h-[22rem] space-y-2 overflow-y-auto pr-0.5">

          {turns
            .map((t, i) => ({ t, i }))
            .reverse()
            .map(({ t, i }) => {
              const diffs = t.ops.flatMap((op) => opDiffs(op, segById, speakers));
              const armed = confirming === i;
              return (
                <div
                  key={i}
                  className="rounded-lg border border-line bg-surface/60 px-3 py-2"
                >
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="text-[12.5px] text-ink">{t.message}</div>
                      <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-ink4">
                        <span
                          className={`inline-block rounded px-1.5 py-px text-[10px] font-medium ${
                            t.source === 'script'
                              ? 'bg-emerald-500/15 text-emerald-700'
                              : 'bg-violet-500/15 text-violet-700'
                          }`}
                        >
                          {t.source === 'script' ? 'user' : 'agent'}
                        </span>
                        <span>{sourceLabel(t.source)}</span>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        if (armed) onDrop(i);
                        else setConfirming(i);
                      }}
                      onBlur={() => setConfirming((c) => (c === i ? null : c))}
                      disabled={busy}
                      aria-label={
                        armed
                          ? `Confirm removing: ${t.message}`
                          : `Remove staged change: ${t.message}`
                      }
                      title={armed ? 'Click again to remove' : 'Remove this change'}
                      className={`flex flex-none items-center justify-center gap-1 self-start
                                  rounded-lg px-2.5 py-1.5 text-[12px] font-medium text-white
                                  transition-colors disabled:opacity-50 ${
                                    armed
                                      ? 'bg-danger-fill ring-1 ring-danger'
                                      : 'bg-danger-fill/85 hover:bg-danger-fill'
                                  }`}
                    >
                      <span aria-hidden="true">🗑</span>
                      {armed ? <span>Remove?</span> : null}
                    </button>
                  </div>

                  {t.ops.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {t.ops.map((op, k) => {
                        const kind = String(
                          (op as Record<string, unknown>).kind ?? '',
                        );
                        const name = String(
                          (op as Record<string, unknown>).op ?? '',
                        );
                        if (!kind || !name) return null;
                        return (
                          <span
                            key={k}
                            title={kindTitle(kind)}
                            className={`inline-flex items-center gap-1 rounded px-1.5 py-px text-[10px] font-medium ${kindChipClass(kind)}`}
                          >
                            <span className="uppercase tracking-wide">{kind}</span>
                            <span className="font-normal text-ink3">·</span>
                            <span className="font-mono">{name}</span>
                          </span>
                        );
                      })}
                    </div>
                  )}

                  <div className="mt-2 space-y-1.5">
                    {diffs.map((d, k) => (
                      <div key={k} className="rounded-md bg-canvas/50 px-2.5 py-1.5">
                        <div className="font-mono text-[11px] text-ink3">{d.object}</div>
                        {d.before ? (
                          <div className="mt-0.5 text-[12px] text-ink4 line-through">
                            {d.before}
                          </div>
                        ) : null}
                        <div
                          className={`mt-0.5 text-[12.5px] ${
                            d.pending ? 'italic text-ink3' : 'text-ink'
                          }`}
                        >
                          {d.after}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
        </div>
      )}

      {canApply ? (
        <div className="mt-3 space-y-2 border-t border-line/60 pt-3">
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" onClick={() => setReviewOpen(true)} disabled={busy}>
              Review &amp; apply
            </Button>
            <Button variant="ghost" onClick={onDiscard} disabled={busy}>
              Discard edits
            </Button>
            <span className="text-[12px] text-ink3">
              re-runs {dirtyStages.join(' → ')}
              {cost?.total_est_s ? ` · about ${fmtClock(cost.total_est_s)}` : ''}
              {cost?.gpu_required ? ' · uses the GPU' : ''}
            </span>
          </div>

          {(cost?.warnings ?? []).map((w, i) => (
            <p key={i} className="flex items-start gap-1.5 text-[12px] text-warn">
              <span aria-hidden="true">⚠</span>
              <span>{w}</span>
            </p>
          ))}
        </div>
      ) : null}

      <Dialog
        open={reviewOpen}
        onOpenChange={setReviewOpen}
        width="max-w-xl"
        title="Apply these changes and re-render?"
        description="Review what will be applied, then re-render the stages that depend on these edits."
        footer={
          <>
            <Button variant="ghost" onClick={() => setReviewOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button
              variant="primary"
              busy={busy}
              onClick={() => {
                setReviewOpen(false);
                onApply();
              }}
            >
              Apply &amp; re-render
              {cost?.total_est_s ? ` · about ${fmtClock(cost.total_est_s)}` : ''}
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <section>
            <Eyebrow className="mb-2">
              What you changed{turns.length ? ` · ${turns.length}` : ''}
            </Eyebrow>
            <ul className="space-y-1.5">
              {turns.map((t, i) => (
                <li key={i} className="flex items-baseline gap-2 text-[12.5px]">
                  <span
                    className={`inline-block rounded px-1.5 py-px text-[10px] font-medium ${
                      t.source === 'script'
                        ? 'bg-emerald-500/15 text-emerald-700'
                        : 'bg-violet-500/15 text-violet-700'
                    }`}
                  >
                    {t.source === 'script' ? 'user' : 'agent'}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-ink">{t.message}</span>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <Eyebrow className="mb-2">What re-runs</Eyebrow>
            <p className="text-[12.5px] text-ink2">
              {dirtyStages.length ? dirtyStages.join(' → ') : 'Nothing — these changes need no re-render.'}
              {cost?.total_est_s ? ` · about ${fmtClock(cost.total_est_s)}` : ''}
              {cost?.gpu_required ? ' · uses the GPU' : ''}
            </p>
          </section>

          {(cost?.warnings ?? []).length ? (
            <section className="rounded-xl border border-warn/40 bg-warn/[0.07] px-3 py-2.5">
              {(cost?.warnings ?? []).map((w, i) => (
                <p key={i} className="flex gap-2 text-[12.5px] leading-relaxed text-ink2">
                  <span aria-hidden="true">⚠</span>
                  <span>{w}</span>
                </p>
              ))}
            </section>
          ) : null}
        </div>
      </Dialog>
    </section>
  );
}
