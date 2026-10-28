import { useCallback, useEffect, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { SpeakerTable } from '../speakers/SpeakerTable';
import { VideoStage } from '../video/VideoStage';
import { NowPlayingCues } from '../video/NowPlayingCues';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import { fmtClock } from '../../lib/format';
import { ActionWorkspace } from './ActionWorkspace';
import { LipsyncPanel } from './LipsyncPanel';
import { EnhancePanel } from './EnhancePanel';
import { ReviewVideoPanel } from './ReviewVideoPanel';
import { HistoryPanel } from './HistoryPanel';
import type {
  AgentScript,
  AgentScriptRow,
  AgentSpeaker,
  AgentTurn,
  CostPreview,
  ScriptEdit,
} from '../../api/types';

/**
 * Everything you can do to a finished dub, under the video that shows it.
 *
 * This was a separate page at `#/job/<id>/edit` until 2026-10-21. Reviewing a
 * dub therefore meant bouncing between two screens — the video on one, the
 * controls that change it on the other — so judging a voice meant
 * remembering what it sounded like. It is one column now, in the order the
 * work actually happens:
 *
 *   1. the cast          who is speaking, in which voice, in which delivery
 *   2. the chat box      ask for a change in plain language
 *   3. the script        read the dub, correct a line by typing it
 *   4. lip-sync          commit the GPU, once the three above look right
 *   5. things you can say / what it can do
 *
 * `backend/edit_agent.py` turns a sentence into typed ops, and `POST
 * /api/jobs/{id}/agent` STAGES them — nothing re-renders until Apply, which
 * is the existing `edits/apply` route with its cost preview. A misparsed
 * sentence therefore costs nothing, which is the whole reason the agent is
 * not wired straight to the GPU.
 *
 * Messages ACCUMULATE. "make him female" followed by "and speed up the
 * intro" is two edits on one plan, not a correction of the first.
 *
 * There is no "What this can change" summary block. It listed the clip,
 * language, lip-sync model and every speaker's voice and tone — a second,
 * read-only copy of facts the cast table above and the lip-sync panel below
 * already show, kept in step by hand. A staged change now appears ON the
 * thing it changes instead.
 */
/**
 * Fold the live selection into a TYPED request.
 *
 * Clicking a quick action already produced "cue 5: speed this up by 1.2x";
 * typing your own sentence did not, so the selection never left the browser
 * and the agent scoped the edit to the whole dub — while the "Editing cue
 * #5" chip stayed on screen insisting otherwise.
 *
 * The prefix is the vocabulary the agent documents and
 * `edit_agent.resolve_cue_aliases` rewrites into a canonical SEG id, so
 * nothing new has to be understood on the wire. It is added ONLY when the
 * message carries no scope of its own: a sentence that already names a cue,
 * a speaker or a timestamp is left exactly as written, because the words you
 * typed outrank the row you happen to have highlighted.
 */
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

  const muteCue = useCallback(async (segId: string) => {
    setBusy(true);
    setError(null);
    try {
      await api.muteCue(jobId, segId);
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
      await load(); // params + script follow the new plan
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
      pushToast({
        kind: 'success',
        title: 'Applied',
        body: `Re-running: ${(r.dirty_stages ?? []).join(' → ')}`,
      });
      await load();
      // The cast is re-derived by the replay, so the table's own payload is
      // stale from here on.
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

  const staged = (params?.dirty_stages?.length ?? 0) > 0;

  return (
    <div className="space-y-4">
      {/* ---- top row: the dub on the left, the cues and cast on the right ----

          The video, the chat box and the staged edits stack down the LEFT —
          you describe a change right under the thing it changes. The cues that
          follow playback and the cast sit on the RIGHT, the two read-mostly
          panels. On a narrow screen the grid collapses to one column and the
          same top-to-bottom order holds. */}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        {/* LEFT: player → chat → staged edits */}
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

        {/* RIGHT: the line playing now → the cast */}
        <div className="min-w-0 space-y-4">
          <NowPlayingCues
            segments={script?.segments ?? []}
            busy={busy}
            onMute={(segId) => void muteCue(segId)}
            onStageScript={(edits) => void stageScript(edits)}
          />
          <SpeakerTable scroll segments={script?.segments ?? []} />
        </div>
      </div>

      {/* ---- finish the job ---- */}
      <LipsyncPanel jobId={jobId} onStarted={() => void load()} />
      {/* 9i sits under 9h: it restores whatever that produced, and is
          disabled by the server until there is a render to restore. */}
      <EnhancePanel jobId={jobId} />
      {/* The ASD review video: a read-off-disk overlay of who the run thought
          was speaking. A download, not inline playback — the renderer writes
          mp4v, which browsers will not play. */}
      <ReviewVideoPanel jobId={jobId} />
      <HistoryPanel />
    </div>
  );
}

/** One concrete before/after inside a staged turn. */
interface Diff {
  object: string;
  before: string;
  after: string;
  /** The result only exists once the LLM/render runs, so "after" is a
      promise, not a value — shown in a quieter style. */
  pending?: boolean;
}

/**
 * Turn an agent op (opaque `{op, scope, params}`) into the one or more
 * before/after rows a human can read. The "before" is looked up from the
 * current script and cast so the card shows what the dub says NOW, not just
 * what the op asked for. Defensive about param key names — small models nest
 * them inconsistently, and an unrecognised op still names itself rather than
 * vanishing.
 */
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
    case 'retranslate':
      return [{ object: tgt, before: firstText, after: 're-translated on apply', pending: true }];
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

/**
 * The staged edits, newest first, as before/after cards — the committed plan,
 * directly under the chat box that builds it.
 *
 * Each card is one turn (one thing asked for, in the user's words), broken
 * into the concrete objects it changes and what each reads as before and
 * after. Removal is two-step and per card: the first press arms the question,
 * the second drops that turn and the server rebuilds the rest of the plan.
 *
 * The Apply bar lives here, with the edits it applies, rather than up in the
 * chat box — nothing re-renders until it is pressed.
 */
function sourceLabel(source: string): string {
  switch (source) {
    case 'script': return 'direct edit';
    case 'llm': return 'parsed by LLM';
    case 'rules': return 'parsed by rules';
    default: return source;
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
  // One armed remove at a time, keyed by the turn's real index. Pressing
  // remove on another card moves the question rather than arming two.
  const [confirming, setConfirming] = useState<number | null>(null);

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
          Changes appear here as before → after cards, newest on
          top. Nothing re-renders until you apply.
        </p>
      ) : (
        <div className="max-h-[22rem] space-y-2 overflow-y-auto pr-0.5">
          {/* Newest first, but keep each turn's REAL index for removal. */}
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
            <Button variant="primary" onClick={onApply} disabled={busy}>
              Apply edits
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
          {/* The compute cost a re-render actually incurs, stated before the
              click rather than discovered after it. */}
          {(cost?.warnings ?? []).map((w, i) => (
            <p key={i} className="flex items-start gap-1.5 text-[12px] text-warn">
              <span aria-hidden="true">⚠</span>
              <span>{w}</span>
            </p>
          ))}
        </div>
      ) : null}
    </section>
  );
}
