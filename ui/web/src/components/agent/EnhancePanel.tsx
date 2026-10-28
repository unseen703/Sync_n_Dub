import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconSpinner } from '../ui/Icons';
import * as api from '../../api';
import type { EnhanceReadiness, EnhanceStatus } from '../../api';

/**
 * Stage 9i — "Enhance resolution" on a finished render.
 *
 * Disabled until lip-sync has produced something, which is the server's
 * call: `can_start` and `blockers` come from `GET .../enhance/readiness`
 * rather than being re-derived here. A browser re-deriving the rule is a
 * second implementation that drifts from what the POST enforces.
 *
 * It is NOT tied to a lip-sync model. Restoration takes a video and gives
 * back a video; the mask that limits it to the frames lip-sync changed is
 * built by diffing the render against its input, so it works for wav2lip,
 * musetalk and latentsync alike. Where a job holds several renders —
 * `lipsync/` and `lipsync_asd/` after an A/B — the operator picks which.
 */
const POLL_MS = 4000;

function fmtEstimate(band: [number, number] | null | undefined): string | null {
  if (!band) return null;
  const m = (s: number) => Math.max(1, Math.round(s / 60));
  const [lo, hi] = band;
  return m(lo) === m(hi) ? `about ${m(lo)} min` : `about ${m(lo)}–${m(hi)} min`;
}

export function EnhancePanel({ jobId }: { jobId: string }) {
  const [ready, setReady] = useState<EnhanceReadiness | null>(null);
  const [status, setStatus] = useState<EnhanceStatus | null>(null);
  const [render, setRender] = useState<string>('');
  const [gated, setGated] = useState(true);
  const [strength, setStrength] = useState(1.0);
  const [scale, setScale] = useState<'same' | '2x'>('same');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.getEnhanceReadiness(jobId, render || undefined);
      setReady(r);
      setStatus(r.status);
      if (!render && r.render_id) setRender(r.render_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [jobId, render]);

  useEffect(() => { void load(); }, [load]);

  // Poll only while something is running. A finished job should not keep a
  // timer alive — the job page is left open for long stretches.
  useEffect(() => {
    if (status?.state !== 'running') {
      if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
      return;
    }
    timer.current = window.setInterval(() => {
      void api.getEnhanceStatus(jobId).then(setStatus).catch(() => {});
    }, POLL_MS);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [jobId, status?.state]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await api.startEnhance(jobId, {
        render: render || undefined, gated, strength, scale,
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const running = status?.state === 'running';
  const blocked = !ready?.can_start || running;
  const est = fmtEstimate(gated ? ready?.estimate_s : ready?.estimate_s_ungated);

  return (
    <section className="rounded-lg border border-line bg-surface2 p-4">
      <Eyebrow>Enhance resolution</Eyebrow>
      <p className="mt-1 text-[12.5px] text-ink3">
        Rebuilds facial detail on the finished render with GFPGAN. A second
        pass, not a cheaper one — it costs more than the lip-sync it cleans,
        so it is a separate step you ask for.
      </p>

      {ready?.frames ? (
        <p className="mt-2 text-[12px] text-ink4">
          {ready.width}×{ready.height} · {ready.frames} frames
          {est ? <> · {est}</> : null}
        </p>
      ) : null}

      {/* Which render. Shown only when there is a real choice to make. */}
      {ready && ready.renders.length > 1 ? (
        <label className="mt-3 block text-[12px] text-ink3">
          Render to enhance
          <select
            className="mt-1 w-full rounded border border-line bg-surface px-2 py-1
                       text-[12.5px] text-ink1"
            value={render}
            disabled={running}
            onChange={(e) => setRender(e.target.value)}
          >
            {ready.renders.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </label>
      ) : null}

      <div className="mt-3 space-y-2 text-[12.5px] text-ink2">
        <label className="flex items-start gap-2">
          <input type="checkbox" className="mt-0.5" checked={gated}
                 disabled={running}
                 onChange={(e) => setGated(e.target.checked)} />
          <span>
            Only the frames lip-sync changed
            <span className="block text-[11.5px] text-ink4">
              Much faster, and it leaves every other frame as the source had
              it. Turn off to restore the whole clip.
            </span>
          </span>
        </label>

        <label className="flex items-center gap-2">
          <span className="w-20">Strength</span>
          <input type="range" min={0} max={1} step={0.1} value={strength}
                 disabled={running} className="flex-1"
                 onChange={(e) => setStrength(Number(e.target.value))} />
          <span className="w-8 text-right text-ink3">{strength.toFixed(1)}</span>
        </label>

        <label className="flex items-center gap-2">
          <span className="w-20">Output size</span>
          <select
            className="rounded border border-line bg-surface px-2 py-1 text-ink1"
            value={scale} disabled={running}
            onChange={(e) => setScale(e.target.value as 'same' | '2x')}
          >
            <option value="same">Same as render</option>
            <option value="2x">2× upscale</option>
          </select>
        </label>
        {scale === '2x' ? (
          <p className="flex items-start gap-1.5 text-[11.5px] text-ink4">
            <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
            <span>
              Upscaling also runs Real-ESRGAN over the background, or the face
              would be sharper than the frame around it. Much slower.
            </span>
          </p>
        ) : null}
      </div>

      {/* The server's reasons, verbatim — not a guess made here. */}
      {ready && ready.blockers.length ? (
        <ul className="mt-3 space-y-1">
          {ready.blockers.map((b) => (
            <li key={b} className="flex items-start gap-1.5 text-[12px] text-ink3">
              <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
              <span>{b}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {running ? (
        <p className="mt-3 flex items-center gap-2 text-[12px] text-ink3">
          <IconSpinner /> Enhancing…{est ? ` (${est})` : ''}
        </p>
      ) : null}

      {status?.state === 'done' ? (
        <p className="mt-3 text-[12px] text-ink3">
          Done in {Math.round(status.wall_s ?? 0)}s — restored{' '}
          {status.restored ?? 0} of {status.frames ?? 0} frames
          {status.passed_through
            ? `, ${status.passed_through} passed through`
            : null}.
        </p>
      ) : null}

      {status?.state === 'failed' ? (
        <p className="mt-2 text-[12px] text-warn">{status.error}</p>
      ) : null}
      {error ? <p className="mt-2 text-[12px] text-warn">{error}</p> : null}

      <div className="mt-3 flex items-center gap-2">
        <Button variant="primary" onClick={() => void start()}
                disabled={busy || blocked}>
          {busy || running ? <IconSpinner /> : null} Enhance resolution
        </Button>
        {status?.state === 'done' ? (
          <span className="text-[11.5px] text-ink4">Re-runs over the render.</span>
        ) : null}
      </div>
    </section>
  );
}
