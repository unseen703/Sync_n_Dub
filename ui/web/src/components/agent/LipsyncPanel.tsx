import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconSpinner } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import type {
  EnhanceReadiness, EnhanceStatus, LipsyncReadiness,
} from '../../api/types';

const ENHANCE_POLL_MS = 4000;

/** Fixed, not offered: the one read of "clean this up" a customer needs is
 *  "yes" or "no". Strength, the changed-frames-only gate and output scale
 *  were a dial for whoever tuned GFPGAN against this pipeline, not a choice
 *  a viewer should be asked to make per job. */
const ENHANCE_OPTIONS = { strength: 0.5, gated: true, scale: 'same' as const };

function fmtEstimate(band: [number, number] | null | undefined): string | null {
  if (!band) return null;
  const m = (s: number) => Math.max(1, Math.round(s / 60));
  const [lo, hi] = band;
  return m(lo) === m(hi) ? `about ${m(lo)} min` : `about ${m(lo)}–${m(hi)} min`;
}

/** Lip-sync controls, plus "Enhance resolution" as a disabled-until-rendered
 *  row below the model choices (folded in from the standalone EnhancePanel
 *  on 2026-10-28 — it had no other caller). Input: job id and optional
 *  start callback. Output: readiness details and a render action. */
export function LipsyncPanel({ jobId, onStarted }: {
  jobId: string;
  onStarted?: () => void;
}) {
  const pushToast = useStore((s) => s.pushToast);
  const [ready, setReady] = useState<LipsyncReadiness | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [confirmDiscard, setConfirmDiscard] = useState(false);

  // Hard-pinned: the UI only offers LatentSync 1.5 now, so there is no
  // selection to make. MuseTalk and Wav2Lip are still in the backend
  // registry for CLI use.
  const LIPSYNC_MODEL = 'latentsync15';

  // -- enhance (GFPGAN), folded in as a row below the model choices --------
  // Disabled until a lip-sync render exists; `enhance/readiness` already
  // says so via `blockers`, the same server-truth contract as the models
  // above. The render it targets is pinned to the plain `*.lipsync.mp4` —
  // never a `.windowed.mp4`/`.remux.mp4` by-product, which `renders` also
  // lists and which the server's own default pick is not guaranteed to skip.
  const [enh, setEnh] = useState<EnhanceReadiness | null>(null);
  const [enhStatus, setEnhStatus] = useState<EnhanceStatus | null>(null);
  const [enhBusy, setEnhBusy] = useState(false);
  const [enhError, setEnhError] = useState<string | null>(null);
  const enhTimer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      setReady(await api.getLipsyncReadiness(jobId));
      setError(null);
    } catch (e) {

      setReady(null);
      setError((e as Error).message);
    }
  }, [jobId]);

  const loadEnhance = useCallback(async () => {
    try {
      const first = await api.getEnhanceReadiness(jobId);
      const plain = first.renders.find((id) => id.endsWith('.lipsync.mp4'));
      const r = plain && plain !== first.render_id
        ? await api.getEnhanceReadiness(jobId, plain)
        : first;
      setEnh(r);
      setEnhStatus(r.status);
      setEnhError(null);
    } catch (e) {
      setEnh(null);
      setEnhError((e as Error).message);
    }
  }, [jobId]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => { void loadEnhance(); }, [loadEnhance]);

  useEffect(() => {
    if (enhStatus?.state !== 'running') {
      if (enhTimer.current) { window.clearInterval(enhTimer.current); enhTimer.current = null; }
      return;
    }
    enhTimer.current = window.setInterval(() => {
      void api.getEnhanceStatus(jobId).then(setEnhStatus).catch(() => {});
    }, ENHANCE_POLL_MS);
    return () => { if (enhTimer.current) window.clearInterval(enhTimer.current); };
  }, [jobId, enhStatus?.state]);

  const enhWasRunning = useRef(false);
  useEffect(() => {
    const st = enhStatus?.state;
    if (st === 'running') { enhWasRunning.current = true; return; }
    if (st !== 'done' || !enhWasRunning.current) return;
    enhWasRunning.current = false;
    void useStore.getState().refreshMedia();
    pushToast({
      kind: 'success',
      title: 'The enhanced version is ready',
      body: 'It is now the default above the player. Switch back to compare it with the render it cleaned.',
      action: {
        label: 'Watch it now',
        run: () => useStore.getState().setVideoTrack('enhanced'),
      },
    });
  }, [enhStatus?.state, pushToast]);

  const startEnhance = async () => {
    const render = enh?.renders.find((id) => id.endsWith('.lipsync.mp4')) ?? enh?.render_id;
    if (!render) return;
    setEnhBusy(true);
    setEnhError(null);
    try {
      setEnhStatus(await api.startEnhance(jobId, { render, ...ENHANCE_OPTIONS }));
    } catch (e) {
      setEnhError((e as Error).message);
    } finally {
      setEnhBusy(false);
    }
  };

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.startLipsync(jobId, LIPSYNC_MODEL);
      pushToast({
        kind: 'success',
        title: `Lip-sync queued — ${res.model}`,
        body: `Replays ${res.dirty_stages.join(' → ')}.`,
      });
      onStarted?.();

      void useStore.getState().refreshProgress();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const discardAndContinue = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.discardEdits(jobId);
      setConfirmDiscard(false);
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
        <Eyebrow className="mb-2">Lipsync and Enhance</Eyebrow>
        <p className="text-[12px] text-ink4">{error}</p>
      </section>
    );
  }

  const lipBlocked = (ready ? !ready.can_start : true) || busy;
  const lipEstimate = ready?.estimates?.[LIPSYNC_MODEL];

  // Disabled until there is a lip-sync render to clean up — `enh` is null
  // before the first readiness response, same "not yet known" start as
  // `lipBlocked` above, so it reads as disabled rather than as a stale enabled
  // button during that gap.
  const enhRunning = enhStatus?.state === 'running';
  const enhBlocked = !enh?.can_start || enhBusy || enhRunning;
  const enhEst = fmtEstimate(enh?.estimate_s);

  return (
    <section id="lipsync-panel" className="card p-4">
      <Eyebrow className="mb-3">Lipsync and Enhance</Eyebrow>

      <div className="space-y-1.5">
        {/* Lip-sync as a non-selectable card, mirroring the Enhance row below.
            One model (LatentSync 1.5), one action — no radio, no choose-a-model
            prompt. */}
        <div
          className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2
                      text-[12.5px] border-line bg-surface/40
                      ${lipBlocked && !busy ? 'opacity-60' : ''}`}
        >
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="font-medium text-ink">
              LipSync 
            </span>
            <span className="text-[11px] leading-snug text-ink3">
              Lip-syncs the dubbed audio to the speaker's mouth.
            </span>
            {lipEstimate ? (
              <span className="text-[11px] text-ink4">
                Expected time {lipEstimate} (5× clip length)
              </span>
            ) : null}
            {ready?.blockers?.length ? (
              <span className="text-[11px] text-ink4">{ready.blockers[0]}</span>
            ) : null}
            {error ? <span className="text-[11px] text-warn">{error}</span> : null}
          </div>
          <Button
            size="sm"
            busy={busy}
            onClick={() => void start()}
            disabled={lipBlocked}
          >
            {busy ? <IconSpinner /> : null} Start lip-sync
          </Button>
        </div>

        {/* Not a choice among the models above — an action on whichever one
            already rendered. Kept in the same list so it reads as "one more
            thing you can do here" rather than a second screen, and disabled
            until there is something to clean up. */}
        <div
          className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2
                      text-[12.5px] ${
                        enhStatus?.state === 'done'
                          ? 'border-ok/35 bg-ok/10'
                          : 'border-line bg-surface/40'
                      } ${enhBlocked && !enhBusy && !enhRunning ? 'opacity-60' : ''}`}
        >
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="font-medium text-ink">Enhance resolution</span>
            <span className="text-[11px] leading-snug text-ink3">
              Restores facial detail on the lip-synced render.
            </span>
            {enh?.frames ? (
              <span className="text-[11px] text-ink4">
                {enh.width}×{enh.height} · {enh.frames} frames
                {enhEst ? ` · ${enhEst}` : ''}
              </span>
            ) : null}
            {!enhRunning && enh?.blockers?.length ? (
              <span className="text-[11px] text-ink4">{enh.blockers[0]}</span>
            ) : null}
            {enhStatus?.state === 'done' ? (
              <span className="text-[11px] text-ink3">
                Restored {enhStatus.restored ?? 0} of {enhStatus.frames ?? 0} frames
                {enhStatus.passed_through ? `, ${enhStatus.passed_through} passed through` : ''}.
              </span>
            ) : null}
            {enhStatus?.state === 'failed' ? (
              <span className="text-[11px] text-warn">{enhStatus.error}</span>
            ) : null}
            {enhError ? <span className="text-[11px] text-warn">{enhError}</span> : null}
          </div>
          <Button
            size="sm"
            busy={enhBusy || enhRunning}
            onClick={() => void startEnhance()}
            disabled={enhBlocked}
          >
            {enhRunning ? `Enhancing${enhEst ? ` (${enhEst})` : ''}` : 'Enhance'}
          </Button>
        </div>
      </div>

      {/* Pending-edit discard still belongs below — it is the one thing a
          viewer may need to resolve before the Start button becomes active. */}
      {ready?.blockers?.length && (ready.pending_edits ?? 0) > 0 ? (
        <div className="mt-3 flex flex-wrap items-center gap-2">
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
    </section>
  );
}
