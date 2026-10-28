import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconSpinner } from '../ui/Icons';
import * as api from '../../api';
import type { ReviewVideoStatus } from '../../api';

/**
 * The ASD review ("bbox") video: who the pipeline thought was speaking, drawn
 * on the clip — one box per tracked face per frame, green for the speaker the
 * render used, red for the rest, grey where nothing is known.
 *
 * It is read off disk (`faces/*.npz` + `asd/*.assignments.json`), never a
 * fresh detection pass, so it shows what the run decided rather than what
 * today would decide. The render is a full decode + composite — minutes at
 * 1080p — so the POST returns at once and this polls the GET, the same shape
 * EnhancePanel uses.
 *
 * Download, not inline playback: the renderer writes mp4v (`cv2.VideoWriter`
 * has no H.264 in this build), which VLC/mpv play but browsers will not. So we
 * hand the operator the file rather than a `<video>` that stays blank.
 */
const POLL_MS = 3000;

export function ReviewVideoPanel({ jobId }: { jobId: string }) {
  const [status, setStatus] = useState<ReviewVideoStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      setStatus(await api.getReviewVideo(jobId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [jobId]);

  useEffect(() => { void load(); }, [load]);

  // Poll only while a render is running — the job page stays open for a long
  // time and a finished render should not keep a timer alive.
  useEffect(() => {
    if (status?.state !== 'running') {
      if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
      return;
    }
    timer.current = window.setInterval(() => {
      void api.getReviewVideo(jobId).then(setStatus).catch(() => {});
    }, POLL_MS);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [jobId, status?.state]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await api.makeReviewVideo(jobId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const running = status?.state === 'running';
  const done = status?.state === 'done' && status.ready;
  const total = status?.frames_total ?? 0;
  const pct = running && total
    ? Math.min(100, Math.round(((status?.frames_done ?? 0) / total) * 100))
    : null;

  return (
    <section className="rounded-lg border border-line bg-surface2 p-4">
      <Eyebrow>ASD review video</Eyebrow>
      <p className="mt-1 text-[12.5px] text-ink3">
        Draws the active-speaker boxes on the clip — green for the face the
        render treated as speaking, red for the rest — straight from the run's
        own ASD output. A review artifact: nothing downstream reads it.
      </p>

      {running ? (
        <p className="mt-3 flex items-center gap-2 text-[12px] text-ink3">
          <IconSpinner /> Rendering…{pct !== null ? ` ${pct}%` : ''}
          {total ? ` (${status?.frames_done ?? 0}/${total} frames)` : ''}
        </p>
      ) : null}

      {status?.state === 'failed' ? (
        <p className="mt-2 flex items-start gap-1.5 text-[12px] text-warn">
          <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
          <span>{status.error ?? 'Render failed.'}</span>
        </p>
      ) : null}
      {error ? <p className="mt-2 text-[12px] text-warn">{error}</p> : null}

      <div className="mt-3 flex items-center gap-2">
        <Button variant="primary" onClick={() => void start()}
                disabled={busy || running}>
          {busy || running ? <IconSpinner /> : null}{' '}
          {done ? 'Re-generate' : 'Generate review video'}
        </Button>
        {done && status.url ? (
          <a href={status.url} download
             className="inline-flex items-center rounded border border-line
                        bg-surface px-3 py-1.5 text-[12.5px] text-ink1
                        hover:bg-surface2">
            Download mp4
          </a>
        ) : null}
      </div>

      {done ? (
        <p className="mt-2 text-[11.5px] text-ink4">
          mp4v codec — plays in VLC or mpv, not in the browser.
        </p>
      ) : null}
    </section>
  );
}
