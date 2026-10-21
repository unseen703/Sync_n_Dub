import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconSparkle, IconSpinner } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import { fmtClock } from '../../lib/format';
import { LipsyncPanel } from './LipsyncPanel';
import type {
  AgentAction,
  AgentParams,
  AgentPrompt,
  AgentScript,
  AgentTurn,
  ScriptEdit,
} from '../../api/types';

/**
 * Edit a finished dub in plain language.
 *
 * The chat box is the only input. `backend/edit_agent.py` turns a sentence
 * into typed ops, and `POST /api/jobs/{id}/agent` STAGES them — nothing
 * re-renders until Apply, which is the existing `edits/apply` route with
 * its cost preview. A misparsed sentence therefore costs nothing, which is
 * the whole reason the agent is not wired straight to the GPU.
 *
 * Messages ACCUMULATE. "make him female" followed by "and speed up the
 * intro" is two edits on one plan, not a correction of the first.
 *
 * The panel shows every parameter the agent can move, with `current` and
 * `staged` side by side, so a sentence's effect is visible before it costs
 * anything. `language` is deliberately read-only: re-translating into a
 * different language is a new run from stage 9a, not an edit.
 */
export function EditAgentPage({ jobId }: { jobId: string }) {
  const pushToast = useStore((s) => s.pushToast);
  const [params, setParams] = useState<AgentParams | null>(null);
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

  const load = useCallback(async () => {
    const [p, s] = await Promise.all([
      api.getAgentParams(jobId),
      api.getAgentScript(jobId),
    ]);
    setParams(p);
    setScript(s);
  }, [jobId]);

  const dirty = Object.keys(drafts);

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
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const staged = (params?.dirty_stages?.length ?? 0) > 0;

  return (
    <div className="mx-auto w-full max-w-6xl px-1 py-6">
      <div className="flex items-center gap-2">
        <IconSparkle width={16} height={16} className="text-brand-ink" />
        <h1 className="text-[15px] font-semibold text-ink">Edit with the agent</h1>
        {/* Sets the store DIRECTLY as well as the hash. Relying on the hash
            alone made this a no-op whenever the URL already read
            `#/job/<id>` — no `hashchange`, no handler, no exit from the
            agent page. The href stays so it is still a real link: middle
            click and "open in new tab" keep working. */}
        <a
          href={`#/job/${encodeURIComponent(jobId)}`}
          onClick={(e) => {
            if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
            e.preventDefault();
            const target = `#/job/${encodeURIComponent(jobId)}`;
            if (location.hash !== target) location.hash = target;
            useStore.getState().setEditing(false);
          }}
          className="ml-auto text-[12.5px] text-ink3 hover:text-ink"
        >
          Back to the dub
        </a>
      </div>

      <div className="mt-4 grid items-start gap-3 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="min-w-0">

      {/* ---- what the agent can change ---- */}
      <section className="card mt-4 p-4">
        <Eyebrow className="mb-3">What this can change</Eyebrow>
        <dl className="grid gap-x-6 gap-y-2 text-[12.5px] sm:grid-cols-2">
          <Row
            label="Clip"
            current={
              params
                ? `${fmtClock(params.clip.current.duration_s ?? 0)} · ${params.clip.current.n_segments} cues`
                : '…'
            }
          />
          <Row
            label="Language"
            current={params?.language.current ?? '…'}
            note={params?.language.note}
          />
          <Row
            label="Lip-sync model"
            current={params?.lipsync.current ?? 'off'}
            stagedValue={params?.lipsync.staged}
          />
          <Row
            label="Lip-sync window"
            current={
              params?.lipsync.window.staged
                ? `${fmtClock(params.lipsync.window.staged.start_s)}–${fmtClock(
                    params.lipsync.window.staged.end_s,
                  )}`
                : 'whole clip'
            }
            stagedValue={params?.lipsync.window.staged ? 'changed' : null}
          />
        </dl>

        <div className="mt-3 border-t border-line/60 pt-3">
          <Eyebrow className="mb-2">Speakers</Eyebrow>
          <ul className="space-y-1.5">
            {(params?.speakers ?? []).map((sp) => (
              <li key={sp.id} className="flex flex-wrap items-center gap-2 text-[12.5px]">
                <span className="font-mono text-[11.5px] text-ink3">{sp.id}</span>
                {sp.gender ? <span className="text-ink4">{sp.gender}</span> : null}
                <span className="text-ink2">
                  voice <Val current={sp.voice.current} staged={sp.voice.staged} />
                </span>
                <span className="text-ink2">
                  tone <Val current={sp.tone.current ?? 'neutral'} staged={sp.tone.staged} />
                </span>
              </li>
            ))}
            {params && params.speakers.length === 0 ? (
              <li className="text-[12.5px] text-ink4">No speakers recorded yet.</li>
            ) : null}
          </ul>
        </div>
      </section>

      {/* ---- the chat ---- */}
      <section className="card mt-3 p-4">
        <Eyebrow className="mb-2">Ask for a change</Eyebrow>
        <div className="max-h-56 space-y-2 overflow-y-auto">
          {(params?.turns ?? []).map((t: AgentTurn, i: number) => (
            <div key={i} className="rounded-lg border border-line bg-surface/60 px-3 py-2">
              <div className="text-[12.5px] text-ink">{t.message}</div>
              <div className="mt-1 text-[11.5px] text-ink4">
                {t.ops.length} op{t.ops.length === 1 ? '' : 's'} · parsed by {t.source}
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>

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

      {/* ---- the script ---- */}
      <section className="card mt-3 p-4">
        <div className="mb-2 flex items-baseline gap-3">
          <Eyebrow>Script{script ? ` · ${script.total} cues` : ''}</Eyebrow>
          <span className="text-[11px] text-ink4">
            Click a line to rewrite it — nothing re-renders until you apply.
          </span>
        </div>

        {dirty.length ? (
          <div className="mb-2 flex items-center gap-3 rounded-lg bg-amber-400/10 px-3 py-2 ring-1 ring-amber-400/40">
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
          {(script?.segments ?? []).map((r) => (
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
                    defaultValue={drafts[r.id] ?? r.text}
                    onBlur={(e) => {
                      const next = e.currentTarget.value.trim();
                      setEditing(null);
                      setDrafts((d) => {
                        const copy = { ...d };
                        // Only a REAL change is kept. Clicking into a line
                        // and out again must not stage an identical edit --
                        // that would re-render the dub to produce the same
                        // audio and charge GPU time for it.
                        if (!next || next === r.text) delete copy[r.id];
                        else copy[r.id] = next;
                        return copy;
                      });
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') e.currentTarget.blur();
                      if (e.key === 'Escape') {
                        e.currentTarget.value = drafts[r.id] ?? r.text;
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
                    {drafts[r.id] ?? r.text ?? ''}
                    {!(drafts[r.id] ?? r.text) ? <em className="text-ink4">—</em> : null}
                  </button>
                )}
                {drafts[r.id] !== undefined ? (
                  <span className="ml-auto flex-none text-[11px] text-brand-ink">edited ·
                    <span className="text-ink4"> was {r.text.slice(0, 24)}</span>
                  </span>
                ) : r.staged ? (
                  <span className="ml-auto flex-none text-[11px] text-brand-ink">staged</span>
                ) : null}
              </div>
            </li>
          ))}
        </ol>
        {script && script.segments.length === 0 ? (
          <p className="text-[12.5px] text-ink4">
            No transcript yet — the agent works on a finished dub.
          </p>
        ) : null}
      </section>

        </div>

        {/* ---- right sidebar: finish the job, then what to say ---- */}
        <aside className="space-y-3 lg:sticky lg:top-4">
          {/* First, because it is what this page is FOR: the review exists
              so the operator can commit the GPU knowingly. It reloads the
              params after starting so the staged/pending state is honest. */}
          <LipsyncPanel jobId={jobId} onStarted={() => void load()} />
          <PromptBook
            prompts={params?.prompts ?? []}
            onPick={(t) => setMsg(t)}
          />
          <ActionBook actions={params?.actions ?? []} />
        </aside>
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

function Row({
  label,
  current,
  stagedValue,
  note,
}: {
  label: string;
  current: string;
  stagedValue?: string | null;
  note?: string;
}) {
  return (
    <div>
      <dt className="text-ink4">{label}</dt>
      <dd className="text-ink">
        <Val current={current} staged={stagedValue ?? null} />
        {note ? <div className="mt-0.5 text-[11.5px] text-ink4">{note}</div> : null}
      </dd>
    </div>
  );
}

/** `current` struck through when something is staged over it. */
function Val({ current, staged }: { current: string | null; staged: string | null }) {
  if (!staged || staged === current) {
    return <span className="text-ink">{current ?? '—'}</span>;
  }
  return (
    <span>
      <span className="text-ink4 line-through">{current ?? '—'}</span>
      <span className="mx-1 text-ink4">→</span>
      <span className="font-medium text-brand-ink">{staged}</span>
    </span>
  );
}
