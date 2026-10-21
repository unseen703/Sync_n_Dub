import { useCallback, useEffect, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconSpinner } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import type { LipsyncReadiness } from '../../api/types';

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
  const [ready, setReady] = useState<LipsyncReadiness | null>(null);
  const [model, setModel] = useState<string>('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <section className="card p-4">
      <Eyebrow className="mb-1">Lip-sync</Eyebrow>
      <p className="mb-3 text-[11px] text-ink4">
        The dub is done. This is the expensive part — choose a model.
      </p>

      {/* ---- what ASD measured ---- */}
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

      {/* ---- the choice ---- */}
      <div className="space-y-1.5">
        {(ready?.models ?? []).map((m) => (
          <label
            key={m}
            className={`flex cursor-pointer items-baseline gap-2 rounded-lg border px-3 py-2
                        text-[12.5px] ${
                          model === m
                            ? 'border-brand/50 bg-brand/10'
                            : 'border-line bg-surface/40'
                        }`}
          >
            <input
              type="radio"
              name={`lipsync-model-${jobId}`}
              value={m}
              checked={model === m}
              onChange={() => setModel(m)}
              className="mt-0.5"
            />
            <span className="font-medium text-ink">{m}</span>
            {face?.available && face.suggests === m ? (
              <span className="rounded bg-brand/15 px-1.5 text-[10.5px] text-brand-ink">
                suits this face size
              </span>
            ) : null}
            <span className="ml-auto text-[11.5px] text-ink4">
              {ready?.estimates?.[m] ?? ''}
            </span>
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
