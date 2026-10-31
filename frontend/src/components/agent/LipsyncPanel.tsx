import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../frontend/Button';
import { Eyebrow } from '../frontend/Card';
import { IconFaceScan, IconSparkle, IconSpinner, IconWave } from '../frontend/Icons';
import { CastConfirmDialog } from '../speakers/CastConfirmDialog';
import { useStore } from '../../store/useStore';
import * as api from '../../api';
import type {
  EnhanceReadiness, EnhanceStatus, LipsyncReadiness,
} from '../../api/types';

function Badge({ children, tone = 'muted' }: { children: React.ReactNode; tone?: 'muted' | 'done' }) {
  return (
    <span
      className={`flex-none rounded-full px-2 py-0.5 text-[12px] font-medium ${
        tone === 'done'
          ? 'bg-emerald-500/15 text-emerald-700'
          : 'bg-ink1/10 text-ink3'
      }`}
    >
      {children}
    </span>
  );
}

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
export function LipsyncPanel({ jobId, onStarted, cueCount = 0 }: {
  jobId: string;
  onStarted?: () => void;
  cueCount?: number;
}) {
  const pushToast = useStore((s) => s.pushToast);
  const [ready, setReady] = useState<LipsyncReadiness | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [confirmDiscard, setConfirmDiscard] = useState(false);

  // -- render dubbed audio (cast → voices → TTS → mix), folded in as the
  // first "next step" — the one call-to-action a viewer sees before anything
  // else is possible, and the same gate SpeakerTable used to render inline.
  const table = useStore((s) => s.speakers);
  const staged = useStore((s) => s.staged);
  const applying = useStore((s) => s.applying);
  const awaitingCast = useStore((s) => Boolean(s.progress?.milestones?.awaiting_cast));
  const mixDone = useStore((s) => Boolean(s.progress?.milestones?.mix_done));
  const selected = useStore((s) => s.selected);
  const stageState = useStore((s) => s.progress?.state);
  const stateEditable = stageState !== 'running' && stageState !== 'queued';
  const [confirmCastOpen, setConfirmCastOpen] = useState(false);
  const [rerunning, setRerunning] = useState(false);
  const [rerunMsg, setRerunMsg] = useState<string | null>(null);

  const unsetVoices = table
    ? table.speakers.filter((sp) => {
      const st = staged[sp.id];
      return !(st?.voice !== undefined ? st.voice : sp.voice);
    })
    : [];
  const chosenVoices = table ? table.speakers.length - unsetVoices.length : 0;

  const startRerun = async () => {
    if (!selected) return;
    setRerunning(true);
    setRerunMsg(null);
    try {
      const r = await api.rerunDub(selected);
      setRerunMsg(r.rerun.launched
        ? 'Re-render launched'
        : `Could not launch: ${r.rerun.error ?? 'unknown error'}`);
    } catch (e) {
      setRerunMsg(`Could not launch: ${(e as Error).message}`);
    } finally {
      setRerunning(false);
    }
  };

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
        <Eyebrow className="mb-2">Next steps</Eyebrow>
        <p className="text-[13px] text-ink4">{error}</p>
      </section>
    );
  }

  const lipBlocked = (ready ? !ready.can_start : true) || busy;
  const lipEstimate = ready?.estimates?.[LIPSYNC_MODEL];
  const showLipButton = !lipBlocked || busy;

  // Disabled until there is a lip-sync render to clean up — `enh` is null
  // before the first readiness response, same "not yet known" start as
  // `lipBlocked` above, so it reads as disabled rather than as a stale enabled
  // button during that gap.
  const enhRunning = enhStatus?.state === 'running';
  const enhBlocked = !enh?.can_start || enhBusy || enhRunning;
  const enhEst = fmtEstimate(enh?.estimate_s);
  const showEnhButton = !enhBlocked || enhBusy || enhRunning;

  return (
    <section id="lipsync-panel" className="card p-4">
      <Eyebrow className="mb-3">Next steps</Eyebrow>

      <div className="space-y-1.5">
        {/* Render dubbed audio — the cast gate SpeakerTable used to render
            inline, folded in here as step one of three so the whole pipeline
            reads as a single list of next steps. */}
        {awaitingCast && table ? (
          <div className="rounded-lg border border-brand/30 bg-brand/5 px-3 py-3">
            <div className="flex items-start gap-2.5">
              <span className="mt-0.5 flex-none text-ink3"><IconWave width={18} height={18} /></span>
              <div className="min-w-0 flex-1">
                <div className="font-medium text-ink">Ready to generate?</div>
                <p className="text-[12px] text-ink3">
                  {table.speakers.length} speaker{table.speakers.length === 1 ? '' : 's'}
                  {' · '}{cueCount} cue{cueCount === 1 ? '' : 's'}
                </p>
                <p className="mt-1 text-[12px] leading-snug text-ink3">
                  Generates speech, fits timing and mixes the final dubbed soundtrack.
                </p>
              </div>
            </div>
            <div className="mt-2.5 flex items-center justify-between gap-2">
              <span className="text-[12px] tabular-nums text-ink4">
                {chosenVoices} of {table.speakers.length} voices chosen
              </span>
              <button
                type="button"
                onClick={() => setConfirmCastOpen(true)}
                disabled={applying || unsetVoices.length > 0}
                title={unsetVoices.length
                  ? `${unsetVoices.length} speaker${unsetVoices.length === 1 ? '' : 's'} still need a voice: ${unsetVoices.map((sp) => sp.label || sp.id).join(', ')}`
                  : undefined}
                className="shrink-0 rounded-lg bg-ink px-3.5 py-2 text-[13px] font-semibold
                           text-white transition-colors hover:bg-ink/90 disabled:opacity-60"
              >
                {applying ? 'Casting…' : 'Render dubbed audio'}
              </button>
            </div>
          </div>
        ) : (
          <div
            className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2.5
                        text-[13px] ${
                          mixDone ? 'border-ok/35 bg-ok/10' : 'border-line bg-surface/40 opacity-60'
                        }`}
          >
            <div className="flex min-w-0 items-start gap-2.5">
              <span className="mt-0.5 flex-none text-ink3"><IconWave width={18} height={18} /></span>
              <div className="flex min-w-0 flex-col gap-0.5">
                <span className="font-medium text-ink">Render dubbed audio</span>
                <span className="text-[12px] leading-snug text-ink3">
                  Generates speech, fits timing and mixes the final dubbed soundtrack.
                </span>
                {rerunMsg ? <span className="text-[12px] text-ink3">{rerunMsg}</span> : null}
              </div>
            </div>
            {mixDone ? (
              stateEditable && selected ? (
                <button
                  type="button"
                  disabled={rerunning}
                  onClick={() => void startRerun()}
                  className="flex-none rounded-lg border border-line bg-surface px-3 py-1.5
                             text-[12px] font-medium text-ink hover:bg-surface-hover
                             disabled:opacity-60"
                  title="Re-run TTS and mix for the latest committed edit. Lip-sync is skipped — run it separately."
                >
                  {rerunning ? 'Starting…' : 'Start TTS & mix'}
                </button>
              ) : (
                <Badge tone="done">Rendered</Badge>
              )
            ) : (
              <Badge>Waiting on casting</Badge>
            )}
          </div>
        )}

        {/* Lip-sync as a non-selectable card, mirroring the Enhance row below.
            One model (LatentSync 1.5), one action — no radio, no choose-a-model
            prompt. */}
        <div
          className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2.5
                      text-[12.5px] border-line bg-surface/40
                      ${lipBlocked && !busy ? 'opacity-60' : ''}`}
        >
          <div className="flex min-w-0 items-start gap-2.5">
            <span className="mt-0.5 flex-none text-ink3"><IconFaceScan width={18} height={18} /></span>
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="font-medium text-ink">Lip-sync</span>
              <span className="text-[12px] leading-snug text-ink3">
                Match the speaker's mouth to the dubbed audio.
              </span>
              {lipEstimate ? (
                <span className="text-[12px] text-ink4">
                  Expected time {lipEstimate} (5× clip length)
                </span>
              ) : null}
              {error ? <span className="text-[12px] text-warn">{error}</span> : null}
            </div>
          </div>
          {showLipButton ? (
            <Button
              size="sm"
              busy={busy}
              onClick={() => void start()}
              disabled={lipBlocked}
            >
              {busy ? <IconSpinner /> : null} Start lip-sync
            </Button>
          ) : (
            <Badge>{ready?.blockers?.[0] ?? 'Available after rendering'}</Badge>
          )}
        </div>

        {/* Not a choice among the models above — an action on whichever one
            already rendered. Kept in the same list so it reads as "one more
            thing you can do here" rather than a second screen, and disabled
            until there is something to clean up. */}
        <div
          className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2.5
                      text-[13px] ${
                        enhStatus?.state === 'done'
                          ? 'border-ok/35 bg-ok/10'
                          : 'border-line bg-surface/40'
                      } ${enhBlocked && !enhBusy && !enhRunning ? 'opacity-60' : ''}`}
        >
          <div className="flex min-w-0 items-start gap-2.5">
            <span className="mt-0.5 flex-none text-ink3"><IconSparkle width={18} height={18} /></span>
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="font-medium text-ink">Enhance resolution</span>
              <span className="text-[12px] leading-snug text-ink3">
                Restore facial detail in the lip-synced video.
              </span>
              {enh?.frames ? (
                <span className="text-[12px] text-ink4">
                  {enh.width}×{enh.height} · {enh.frames} frames
                  {enhEst ? ` · ${enhEst}` : ''}
                </span>
              ) : null}
              {enhStatus?.state === 'done' ? (
                <span className="text-[12px] text-ink3">
                  Restored {enhStatus.restored ?? 0} of {enhStatus.frames ?? 0} frames
                  {enhStatus.passed_through ? `, ${enhStatus.passed_through} passed through` : ''}.
                </span>
              ) : null}
              {enhStatus?.state === 'failed' ? (
                <span className="text-[12px] text-warn">{enhStatus.error}</span>
              ) : null}
              {enhError ? <span className="text-[12px] text-warn">{enhError}</span> : null}
            </div>
          </div>
          {showEnhButton ? (
            <Button
              size="sm"
              busy={enhBusy || enhRunning}
              onClick={() => void startEnhance()}
              disabled={enhBlocked}
            >
              {enhRunning ? `Enhancing${enhEst ? ` (${enhEst})` : ''}` : 'Enhance'}
            </Button>
          ) : (
            <Badge>{enh?.blockers?.[0] ?? 'After lip-sync'}</Badge>
          )}
        </div>
      </div>

      <CastConfirmDialog open={confirmCastOpen} onOpenChange={setConfirmCastOpen} />

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
          <span className="text-[12.5px] text-ink4">
            {confirmDiscard
              ? 'This cannot be undone from here.'
              : 'Drops the staged edits so lip-sync can start.'}
          </span>
        </div>
      ) : null}
    </section>
  );
}
