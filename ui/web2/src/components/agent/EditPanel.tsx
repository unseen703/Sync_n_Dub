import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconSpinner } from '../ui/Icons';
import { StagedText, StagedValue } from '../ui/StagedValue';
import { SpeakerTable } from '../speakers/SpeakerTable';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import { fmtClock } from '../../lib/format';
import { LipsyncPanel } from './LipsyncPanel';
import { EnhancePanel } from './EnhancePanel';
import type {
  AgentAction,
  AgentPrompt,
  AgentScript,
  AgentTurn,
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
export function EditPanel({ jobId }: { jobId: string }) {
  const pushToast = useStore((s) => s.pushToast);
  const params = useStore((s) => s.agentParams);
  const refreshParams = useStore((s) => s.refreshAgentParams);
  const refreshSpeakers = useStore((s) => s.refreshSpeakers);
  const [script, setScript] = useState<AgentScript | null>(null);
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  // Hand-edited lines, keyed by segment id, holding ONLY what differs from
  // what is on screen. A sentence to the agent and a typed correction are
  // two ways to reach the same `set_text` op; the difference is that here
  // the user supplies the wording, so nothing is inferred and the quoted
  // cost is the applied cost.
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [editing, setEditing] = useState<string | null>(null);
  // Index of the staged change whose remove button is asking to be
  // confirmed. One at a time: pressing remove on another turn moves the
  // question rather than arming two at once.
  const [confirming, setConfirming] = useState<number | null>(null);

  const load = useCallback(async () => {
    // Params go through the store, because the cast table reads them too —
    // a gender or voice change asked for in the chat box has to light up on
    // the row it is about, and that row is rendered by `SpeakerTable`.
    const [, s] = await Promise.all([
      refreshParams(),
      api.getAgentScript(jobId),
    ]);
    setScript(s);
  }, [jobId, refreshParams]);

  const dirty = Object.keys(drafts);

  /**
   * Drop one staged change and keep the rest.
   *
   * The server rebuilds the remaining plan from the turns already on disk —
   * no sentence is re-parsed, so removing the third of five cannot change
   * how the other four were understood. It also re-prices, because dropping
   * an op changes which stages replay.
   */
  const dropTurn = useCallback(async (index: number) => {
    setBusy(true);
    setError(null);
    try {
      await api.dropStagedTurn(jobId, index);
      setConfirming(null);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [jobId, load]);

  const stageScript = useCallback(async () => {
    const edits: ScriptEdit[] = Object.entries(drafts).map(([segment_id, text]) => ({
      segment_id,
      text,
    }));
    if (!edits.length) return;
    setBusy(true);
    setError(null);
    try {
      const plan = await api.stageScript(jobId, edits);
      // Staged, not applied. The pending bar owns Apply, which is the one
      // call that spends GPU time -- a typo here costs nothing until then.
      setDrafts({});
      setEditing(null);
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
  }, [drafts, jobId, load, pushToast]);

  useEffect(() => {
    void load().catch((e) => setError((e as Error).message));
  }, [load]);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [params?.turns?.length]);

  const send = async () => {
    const text = msg.trim();
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
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const staged = (params?.dirty_stages?.length ?? 0) > 0;

  return (
    <div className="space-y-4">
      {/* ---- 1. the cast: who, in which voice, in which delivery ---- */}
      <SpeakerTable />

      {/* ---- 2. the chat ---- */}
      <section className="card p-4">
        <Eyebrow className="mb-2">Ask for a change</Eyebrow>
        {params?.turns?.length ? (
          <div className="max-h-56 space-y-2 overflow-y-auto">
            {params.turns.map((t: AgentTurn, i: number) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg border border-line
                           bg-surface/60 px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="text-[12.5px] text-ink">{t.message}</div>
                  <div className="mt-1 text-[11.5px] text-ink4">
                    {t.ops.length} op{t.ops.length === 1 ? '' : 's'} · parsed by {t.source}
                  </div>
                </div>
                {/* Two-step, never a one-click delete: the first press turns
                    the control into the question. `window.confirm` would do
                    the same job but blocks the tab and reads as a browser
                    error; this keeps the warning next to the thing it is
                    about. Clicking any other remove resets this one. */}
                <button
                  type="button"
                  onClick={() => {
                    if (confirming === i) void dropTurn(i);
                    else setConfirming(i);
                  }}
                  onBlur={() => setConfirming((c) => (c === i ? null : c))}
                  disabled={busy}
                  aria-label={
                    confirming === i
                      ? `Confirm removing: ${t.message}`
                      : `Remove staged change: ${t.message}`
                  }
                  title={confirming === i ? 'Click again to remove' : 'Remove this change'}
                  /* `danger`, not a raw Tailwind red: it is the project's
                     semantic token, it follows both themes, and its contrast
                     against white text is what `npm run contrast` checks. */
                  className={`flex flex-none items-center justify-center gap-1 self-center
                              rounded-lg px-2.5 py-1.5 text-[12px] font-medium text-white
                              transition-colors disabled:opacity-50 ${
                                confirming === i
                                  ? 'bg-danger-fill ring-1 ring-danger'
                                  : 'bg-danger-fill/85 hover:bg-danger-fill'
                              }`}
                >
                  <span aria-hidden="true">🗑</span>
                  {confirming === i ? <span>Remove?</span> : null}
                </button>
              </div>
            ))}
            <div ref={endRef} />
          </div>
        ) : (
          <p className="mb-2 text-[11.5px] text-ink4">
            Changes you ask for are staged and shown above and below, against
            the thing they change. Nothing re-renders until you apply.
          </p>
        )}

        <div className="mt-2 flex gap-2">
          <input
            value={msg}
            onChange={(e) => setMsg(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) void send();
            }}
            placeholder='e.g. "make SPEAKER_00 female and speed up 0:30 to 0:45"'
            aria-label="Describe the change"
            className="min-w-0 flex-1 rounded-lg border border-edge/70 bg-canvas/60 px-3 py-2
                       text-[13px] text-ink placeholder:text-ink4"
          />
          <Button onClick={() => void send()} disabled={busy || !msg.trim()}>
            {busy ? <IconSpinner /> : null} Send
          </Button>
        </div>

        {error ? (
          <p className="mt-2 flex items-start gap-1.5 text-[12.5px] text-warn">
            <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
            <span>{error}</span>
          </p>
        ) : null}

        {(params?.warnings ?? []).map((w, i) => (
          <p key={i} className="mt-2 text-[12px] text-ink3">
            {w}
          </p>
        ))}

        {staged ? (
          <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-line/60 pt-3">
            <Button variant="primary" onClick={() => void apply()} disabled={busy}>
              Apply &amp; re-render
            </Button>
            <span className="text-[12px] text-ink3">
              re-runs {params?.dirty_stages.join(' → ')}
              {params?.cost?.total_est_s ? ` · about ${fmtClock(params.cost.total_est_s)}` : ''}
            </span>
          </div>
        ) : null}
      </section>

      {/* ---- 3. the script ---- */}
      <section className="card p-4">
        <div className="mb-2 flex items-baseline gap-3">
          <Eyebrow>Script{script ? ` · ${script.total} cues` : ''}</Eyebrow>
          <span className="text-[11px] text-ink4">
            Click a line to rewrite it — nothing re-renders until you apply.
          </span>
        </div>

        {dirty.length ? (
          <div className="mb-2 flex flex-wrap items-center gap-3 rounded-lg bg-amber-400/10 px-3 py-2 ring-1 ring-amber-400/40">
            <span className="text-[12.5px] text-ink">
              {dirty.length} line{dirty.length === 1 ? '' : 's'} rewritten
            </span>
            <span className="text-[11px] text-ink4">
              Replays speech, mix and lip-sync. Translation, transcript and
              speakers stay as they are.
            </span>
            <div className="ml-auto flex gap-2">
              <Button
                variant="ghost"
                onClick={() => {
                  setDrafts({});
                  setEditing(null);
                }}
              >
                Discard
              </Button>
              <Button onClick={stageScript} disabled={busy}>
                {busy ? <IconSpinner /> : null} Review &amp; apply
              </Button>
            </div>
          </div>
        ) : null}
        <ol className="space-y-1">
          {(script?.segments ?? []).map((r) => {
            // What this line READS AS right now: an unsaved draft, else the
            // wording already staged, else what is in the dub. Editing a
            // staged line used to reopen on the ORIGINAL text and compare
            // against it, so a second edit redrew the row unchanged and
            // looked like it had been ignored.
            const shown = drafts[r.id] ?? r.staged_text ?? r.text ?? '';
            // The reading this one REPLACES, for the before/after below. A
            // typed draft is measured against whatever was on screen when
            // the user clicked into it; a server-staged rewrite against the
            // wording in the dub.
            const was = drafts[r.id] !== undefined
              ? (r.staged_text ?? r.text)
              : r.staged_text ? r.text : null;
            return (
              <li
                key={r.id}
                className={`rounded-lg px-2.5 py-1.5 text-[12.5px] ${
                  drafts[r.id] !== undefined
                    ? 'bg-amber-400/10 ring-1 ring-amber-400/40'
                    : r.staged
                      ? 'bg-brand/10 ring-1 ring-brand/30'
                      : 'odd:bg-surface/40'
                }`}
              >
                <div className="flex items-baseline gap-2">
                  <span className="flex-none font-mono text-[11px] text-ink4">
                    {fmtClock(r.start)}
                  </span>
                  {r.speaker ? (
                    <span className="flex-none font-mono text-[11px] text-ink3">{r.speaker}</span>
                  ) : null}
                  {editing === r.id ? (
                    <input
                      autoFocus
                      className="min-w-0 flex-1 rounded border border-brand/40 bg-surface px-1.5 py-0.5 text-ink outline-none"
                      defaultValue={shown}
                      onBlur={(e) => {
                        const next = e.currentTarget.value.trim();
                        const before = r.staged_text ?? r.text;
                        setEditing(null);
                        setDrafts((d) => {
                          const copy = { ...d };
                          // Only a REAL change is kept, and "real" is against
                          // what the line currently reads as. Clicking into a
                          // line and out again must not stage an identical
                          // edit — that would re-render the dub to produce
                          // the same audio and charge GPU time for it.
                          if (!next || next === before) delete copy[r.id];
                          else copy[r.id] = next;
                          return copy;
                        });
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') e.currentTarget.blur();
                        if (e.key === 'Escape') {
                          e.currentTarget.value = shown;
                          e.currentTarget.blur();
                        }
                      }}
                    />
                  ) : (
                    <button
                      type="button"
                      className="min-w-0 flex-1 cursor-text text-left text-ink hover:underline decoration-dotted underline-offset-2"
                      title="Click to edit this line"
                      onClick={() => setEditing(r.id)}
                    >
                      {shown || <em className="text-ink4">—</em>}
                    </button>
                  )}
                  {/* Speed per cue. Shown only when it is not 1.0 or about
                      to change — a column of "1.0×" down every row is
                      noise, and the one cue that differs is the whole
                      point. */}
                  {r.staged_speed != null && r.staged_speed !== r.speed ? (
                    <span className="ml-auto flex-none font-mono text-[11px]">
                      <StagedValue
                        current={`${r.speed.toFixed(2)}×`}
                        staged={`${r.staged_speed.toFixed(2)}×`}
                        label={`Speed of the cue at ${fmtClock(r.start)}`}
                      />
                    </span>
                  ) : r.speed !== 1 ? (
                    <span
                      className="ml-auto flex-none font-mono text-[11px] text-ink3"
                      title="How fast this cue is spoken, against its own slot"
                    >
                      {r.speed.toFixed(2)}×
                    </span>
                  ) : null}
                  {was !== null ? (
                    <span className="flex-none text-[11px] text-blue-ink">
                      {drafts[r.id] !== undefined ? 'edited' : 'staged'}
                    </span>
                  ) : r.staged ? (
                    <span className="flex-none text-[11px] text-blue-ink">staged</span>
                  ) : null}
                </div>
                {/* The change itself, UNDER the line rather than beside it.
                    This used to be a right-aligned `was <first 24 chars>`,
                    which is shorter than most cues — so it hid the very
                    words that were changing, which is the one thing a
                    translation edit has to be judged on. */}
                {was !== null && editing !== r.id ? (
                  <StagedText current={was} staged={shown} />
                ) : null}
              </li>
            );
          })}
        </ol>
        {script && script.segments.length === 0 ? (
          <p className="text-[12.5px] text-ink4">
            No transcript yet — the agent works on a finished dub.
          </p>
        ) : null}
      </section>

      {/* ---- 4. finish the job ---- */}
      <LipsyncPanel jobId={jobId} onStarted={() => void load()} />
      {/* 9i sits under 9h: it restores whatever that produced, and is
          disabled by the server until there is a render to restore. */}
      <EnhancePanel jobId={jobId} />

      {/* ---- 5. reference ---- */}
      <div className="grid gap-4 lg:grid-cols-2">
        <PromptBook prompts={params?.prompts ?? []} onPick={(t) => setMsg(t)} />
        <ActionBook actions={params?.actions ?? []} />
      </div>
    </div>
  );
}

/**
 * Requests the user can copy, three at a time.
 *
 * The list comes from the backend, which builds it from each operation's own
 * worked examples. Writing it here would let the panel advertise a phrasing
 * the parser does not accept -- the failure `ui/speakers.py` already shipped
 * once with an invented tone vocabulary.
 *
 * Clicking one fills the message box rather than sending it: the point is to
 * show the SHAPE, and the times and speaker still have to be the user's.
 */
function PromptBook({
  prompts,
  onPick,
}: {
  prompts: AgentPrompt[];
  onPick: (text: string) => void;
}) {
  const [all, setAll] = useState(false);
  const shown = all ? prompts : prompts.slice(0, 3);
  if (!prompts.length) return null;
  return (
    <section className="card p-4">
      <Eyebrow className="mb-1">Things you can say</Eyebrow>
      <p className="mb-2 text-[11px] text-ink4">
        Click one to put it in the box, then change the times and names.
      </p>
      <ul className="space-y-1">
        {shown.map((p) => (
          <li key={p.text}>
            <button
              type="button"
              onClick={() => onPick(p.text)}
              title={p.does}
              className="w-full rounded-lg bg-surface/50 px-2.5 py-1.5 text-left text-[12px] text-ink
                         hover:bg-surface hover:ring-1 hover:ring-brand/30"
            >
              &ldquo;{p.text}&rdquo;
            </button>
          </li>
        ))}
      </ul>
      {prompts.length > 3 ? (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="mt-2 text-[11.5px] text-ink3 hover:text-ink"
        >
          {all ? 'Show fewer' : `Show ${prompts.length - 3} more`}
        </button>
      ) : null}
    </section>
  );
}

/**
 * What the agent can do, three at a time: the action, what it needs from
 * you, and what it changes about the dub.
 *
 * `replays` is shown as a separate, quieter line because it is a different
 * kind of fact -- the cost, not the effect -- and the backend derives it
 * from the operation's own `dirty_from` so it cannot claim a cheaper
 * re-render than it causes.
 */
function ActionBook({ actions }: { actions: AgentAction[] }) {
  const [all, setAll] = useState(false);
  const shown = all ? actions : actions.slice(0, 3);
  if (!actions.length) return null;
  return (
    <section className="card p-4">
      <Eyebrow className="mb-1">What it can do</Eyebrow>
      <p className="mb-2 text-[11px] text-ink4">
        Nothing re-renders until you apply.
      </p>
      <ul className="space-y-2.5">
        {shown.map((a) => (
          <li key={a.op} className="rounded-lg bg-surface/40 px-2.5 py-2">
            <p className="text-[12.5px] font-medium text-ink">{a.title}</p>
            <p className="mt-0.5 text-[11.5px] text-ink3">
              <span className="text-ink4">You give it </span>
              {a.takes}
            </p>
            <p className="mt-0.5 text-[11.5px] text-ink3">
              <span className="text-ink4">Then </span>
              {a.changes}
            </p>
            <p className="mt-1 text-[11px] text-ink4">Re-runs: {a.replays}</p>
          </li>
        ))}
      </ul>
      {actions.length > 3 ? (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="mt-2 text-[11.5px] text-ink3 hover:text-ink"
        >
          {all ? 'Show fewer' : `Show ${actions.length - 3} more`}
        </button>
      ) : null}
    </section>
  );
}
