import { useCallback, useEffect, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconSpinner } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import type { LipsyncModelOption, LipsyncReadiness } from '../../api/types';

/**
 * The second half of a dub: choose a lip-sync model and spend the GPU.
 *
 * A submission dubs the audio and STOPS. This panel is where the operator,
 * having reviewed the cast and the script on this page, picks a model and
 * starts the render.
 *
 * Three things here are deliberate.
 *
 * **No model is preselected.** The server 422s without one
 * (`A lip-sync model is required to proceed`) rather than defaulting, and
 * preselecting here would put the choice back on the machine by the back
 * door — a default nobody changes is a decision nobody made. The previous
 * build picked the model from a face-size probe the operator never saw.
 *
 * **`can_start` and `blockers` come from the server**, not from re-deriving
 * the rules in the browser. They are produced by the same
 * `_lipsync_blockers()` the POST enforces, so a green button cannot drift
 * into offering an action the server would refuse.
 *
 * **The face size is shown, not acted on.** `face.suggests` is a
 * suggestion beside the number; nothing selects it. It is measured from the
 * track ASD attributed to each speaker, not "the biggest face in frame" —
 * on a multi-face shot those differ, and lip-sync repaints the speaker's
 * mouth.
 */
export function LipsyncPanel({ jobId, onStarted }: {
  jobId: string;
  onStarted?: () => void;
}) {
  const pushToast = useStore((s) => s.pushToast);
  const admin = useStore((s) => s.mode === 'admin');
  const [ready, setReady] = useState<LipsyncReadiness | null>(null);
  const [model, setModel] = useState<string>('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Two-step on the destructive path: discarding staged edits throws away
  // work that cannot be recovered from this screen.
  const [confirmDiscard, setConfirmDiscard] = useState(false);

  const load = useCallback(async () => {
    try {
      setReady(await api.getLipsyncReadiness(jobId));
      setError(null);
    } catch (e) {
      // A job that predates the endpoint, or one still running. Not fatal:
      // the panel just says it cannot tell yet.
      setReady(null);
      setError((e as Error).message);
    }
  }, [jobId]);

  useEffect(() => { void load(); }, [load]);

  const start = async () => {
    if (!model) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.startLipsync(jobId, model);
      pushToast({
        kind: 'success',
        title: `Lip-sync queued — ${res.model}`,
        body: `Replays ${res.dirty_stages.join(' → ')}.`,
      });
      onStarted?.();
      // No navigation any more: this panel is ON the job page, under the
      // player and the progress bar it used to send the operator back to.
      // It used to live on a separate review page, where staying put left
      // them watching a panel for a job that had moved on. Refresh instead,
      // so the ribbon picks the render up on its next tick.
      void useStore.getState().refreshProgress();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  /**
   * Drop the staged edits and carry on to lip-sync.
   *
   * The blocker is correct — rendering around a pending edit would use the
   * audio from before it — but "apply or discard" offered no way to do the
   * second without leaving the page. Discarding is destructive, so it asks
   * first, in the panel rather than in a browser dialog.
   */
  const discardAndContinue = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.discardEdits(jobId);
      setConfirmDiscard(false);
      onStarted?.();        // the page reloads its own staged state
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (error && !ready) {
    return (
      <section className="card p-4">
        <Eyebrow className="mb-2">Lip-sync</Eyebrow>
        <p className="text-[12px] text-ink4">{error}</p>
      </section>
    );
  }

  const face = ready?.face;
  const blocked = ready ? !ready.can_start : true;

  // `model_rows` is the labelled list; `models` is the bare registry ids and
  // is still served, so a page open against an older server keeps working
  // rather than rendering an empty picker.
  const rows: LipsyncModelOption[] =
    ready?.model_rows ?? (ready?.models ?? []).map((id) => ({ id, label: id }));

  return (
    <section id="lipsync-panel" className="card p-4">
      <Eyebrow className="mb-1">Lip-sync</Eyebrow>
      <p className="mb-3 text-[11px] text-ink4">
        The dub is done. This is the expensive part — choose a model.
      </p>

      {/* ---- what ASD measured ----
          Admin only: pixel heights and per-speaker frame counts are a
          measurement, and nothing on this panel acts on them (see the
          "shown, not acted on" note above). Someone choosing a model from
          its label does not need them; someone debugging a bad mouth does.
          The CHOICE itself stays in both modes — the model rows carry
          server-supplied labels already. */}
      {admin ? (
      <div className="mb-3 rounded-lg border border-line bg-surface/60 px-3 py-2">
        <div className="text-[11px] uppercase tracking-wide text-ink4">
          Average face size
        </div>
        {face?.available ? (
          <>
            <div className="mt-0.5 text-[13px] text-ink">
              {face.avg_px}px
              <span className="text-ink3">
                {' '}· {Math.round((face.avg_frac ?? 0) * 100)}% of frame height
              </span>
            </div>
            {face.why ? (
              <p className="mt-1 text-[11.5px] text-ink3">{face.why}</p>
            ) : null}
            {(face.per_speaker ?? []).length > 1 ? (
              <ul className="mt-1.5 space-y-0.5">
                {(face.per_speaker ?? []).map((s) => (
                  <li key={s.speaker} className="text-[11.5px] text-ink3">
                    <span className="font-mono text-[11px] text-ink4">{s.speaker}</span>
                    {' '}{s.avg_px}px over {s.frames} frames
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        ) : (
          <p className="mt-0.5 text-[11.5px] text-ink4">
            {face?.reason ?? 'Not measured for this job.'}
          </p>
        )}
      </div>
      ) : null}

      {/* ---- the choice ---- */}
      <div className="space-y-1.5">
        {rows.map((row) => (
          <label
            key={row.id}
            className={`flex cursor-pointer flex-col gap-1 rounded-lg border px-3 py-2
                        text-[12.5px] ${
                          model === row.id
                            ? 'border-brand/50 bg-brand/10'
                            : 'border-line bg-surface/40'
                        }`}
          >
            <span className="flex items-baseline gap-2">
              <input
                type="radio"
                name={`lipsync-model-${jobId}`}
                value={row.id}
                checked={model === row.id}
                onChange={() => setModel(row.id)}
                className="mt-0.5"
              />
              <span className="font-medium text-ink">{row.label}</span>
              {row.speed ? (
                <span className="text-[11px] text-ink4">{row.speed}</span>
              ) : null}
              {face?.available && face.suggests === row.id ? (
                <span className="rounded bg-brand/15 px-1.5 text-[10.5px] text-brand-ink">
                  suits this face size
                </span>
              ) : null}
              <span className="ml-auto text-[11.5px] text-ink4">
                {ready?.estimates?.[row.id] ?? ''}
              </span>
            </span>
            {/* The caveat belongs in front of the click, not in a log after
                it: LatentSync is the slowest by far and has no measured
                band on this host, so its cost cannot be quoted up front. */}
            {row.note ? (
              <span className="pl-6 text-[11px] leading-snug text-ink3">
                {row.note}
              </span>
            ) : null}
          </label>
        ))}
      </div>

      {ready?.blockers?.length ? (
        <div className="mt-3 space-y-1">
          {ready.blockers.map((b, i) => (
            <p key={i} className="flex items-start gap-1.5 text-[12px] text-warn">
              <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
              <span>{b}</span>
            </p>
          ))}
          {/* The way out of the staged-edits blocker without leaving the
              page. Only offered for THAT blocker: "no dubbed audio yet" is
              not something discarding can fix, and a button that cannot
              help is worse than none. */}
          {(ready.pending_edits ?? 0) > 0 ? (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <button
                type="button"
                onClick={() => {
                  if (confirmDiscard) void discardAndContinue();
                  else setConfirmDiscard(true);
                }}
                onBlur={() => setConfirmDiscard(false)}
                disabled={busy}
                className={`rounded-lg px-2.5 py-1.5 text-[12px] font-medium
                            text-white transition-colors disabled:opacity-50 ${
                              confirmDiscard
                                ? 'bg-danger-fill ring-1 ring-danger'
                                : 'bg-danger-fill/85 hover:bg-danger-fill'
                            }`}
              >
                {confirmDiscard
                  ? `Discard ${ready.pending_edits} edit(s)?`
                  : 'Continue without them'}
              </button>
              <span className="text-[11.5px] text-ink4">
                {confirmDiscard
                  ? 'This cannot be undone from here.'
                  : 'Drops the staged edits so lip-sync can start.'}
              </span>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* The required-model warning, which is the whole point of removing
          the automatic choice. Stated before the click, not after a 422. */}
      {!model && !blocked ? (
        <p className="mt-3 flex items-start gap-1.5 text-[12px] text-ink3">
          <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
          <span>Pick a model to continue — lip-sync will not start without one.</span>
        </p>
      ) : null}

      {error ? (
        <p className="mt-2 text-[12px] text-warn">{error}</p>
      ) : null}

      <div className="mt-3 flex items-center gap-2">
        <Button
          variant="primary"
          onClick={() => void start()}
          disabled={busy || blocked || !model}
        >
          {busy ? <IconSpinner /> : null} Start lip-sync
        </Button>
        {ready?.already_rendered ? (
          <span className="text-[11.5px] text-ink4">Re-renders the existing one.</span>
        ) : null}
      </div>
    </section>
  );
}
