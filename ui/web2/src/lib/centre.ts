/**
 * What the centre column is showing right now.
 *
 * One function, so the reveal order is stated once instead of being spread
 * across six `progress.milestones.x && media.y` conditions that drift.
 *
 * Two rules worth keeping explicit:
 *
 *  - `failed` is checked FIRST and unconditionally, but the player stays
 *    mounted below the banner. A job that dies at lip-sync still has a
 *    perfectly good original video and the user should see both.
 *  - The player appears at `video_ready`, which lands several stages before
 *    diarization — the cut clip exists long before we know who is speaking.
 *    The speaker table is the thing gated on `diarization_done`.
 */
import type { Media, Progress } from '../api/types';

export type CentreState =
  | 'no-job'
  | 'creating'
  | 'preparing'
  | 'source-only'
  | 'dub-ready'
  | 'failed';

export function deriveCentreState(
  progress: Progress | null,
  media: Media | null,
): CentreState {
  if (!progress) return 'no-job';
  if (progress.state === 'failed' || progress.state === 'cancelled') return 'failed';
  if (progress.state === 'queued') return 'creating';

  const m = progress.milestones;
  if (!m?.video_ready || !media?.video.original) return 'preparing';
  if (m.mix_done && media.audio.dubbed) return 'dub-ready';
  return 'source-only';
}

/**
 * Whether the Original ⇄ Lip-synced toggle exists at all, and if so whether
 * its second segment is live.
 *
 * `configured: false` means the job has no lip-sync stage — the toggle is
 * ABSENT, not disabled. A permanently dead control is noise.
 *
 * When it is configured but not finished, the segment renders disabled with
 * its own progress. It never falls back to playing the original under a
 * "Lip-synced" label; that would defeat the whole point of an A/B toggle.
 */
export function lipsyncToggleState(progress: Progress | null, media: Media | null): {
  configured: boolean;
  ready: boolean;
  pct: number;
  running: boolean;
} {
  const stages = progress?.stages ?? [];
  const lip = stages.filter((s) => s.key === '9h' || s.key.startsWith('P6_') || s.key.startsWith('P7_'));
  const configured = lip.length > 0 || Boolean(media?.video.lipsync);
  const ready = Boolean(progress?.milestones?.lipsync_ready && media?.video.lipsync);
  const running = lip.some((s) => s.state === 'running');
  // Stage 9h shells into the musetalk/latentsync conda envs, so its internal
  // progress reaches no logger of ours. The only honest number here is the
  // overall job percentage; the ribbon and the log tail carry the detail.
  return { configured, ready, pct: progress?.pct ?? 0, running };
}

// ---------------------------------------------------------------------------
// The product-level outcome: is the result usable, and what is the next step.
// ---------------------------------------------------------------------------

export type OutcomeKind = 'processing' | 'dub-ready' | 'lipsync-ready' | 'failed';
export type OutcomeTone = 'ok' | 'progress' | 'bad';

export interface OutcomeAction {
  kind: 'sync-lips' | 'download';
  label: string;
  href?: string | null;
}

export interface Outcome {
  kind: OutcomeKind;
  /** The headline, e.g. "Dubbed video ready". */
  status: string;
  /** A second line, e.g. "Lip-sync hasn't been run yet." Null when the
      headline says it all. */
  sub: string | null;
  /** Drives the icon AND the colour, so the state never rests on colour alone. */
  tone: OutcomeTone;
  action: OutcomeAction | null;
  /** The dub download, offered as a secondary link when the headline action
      is sync-lips — the dub is the deliverable even then. */
  dubDownloadUrl: string | null;
}

/**
 * The PRODUCT-level outcome, as opposed to `deriveCentreState`'s pipeline view.
 *
 * The distinction is the whole point: a finished dub with no lip-sync is a
 * usable result, not an incomplete pipeline. This says "Dubbed video ready"
 * where the stage ribbon would say "Lip sync not started".
 *
 * `failed` carries no action — `FailureCard` owns the explanation and the
 * "Start it again" button, and two retry controls on one screen is one too many.
 */
export function deriveOutcome(progress: Progress | null, media: Media | null): Outcome {
  const centre = deriveCentreState(progress, media);
  const lip = lipsyncToggleState(progress, media);
  const dubUrl = media?.video.dubbed_no_lipsync?.download_url ?? null;

  if (centre === 'failed') {
    const cancelled = progress?.state === 'cancelled';
    return {
      kind: 'failed',
      status: cancelled ? 'Dub stopped' : 'Dub failed',
      sub: null,
      tone: 'bad',
      action: null,
      dubDownloadUrl: dubUrl,
    };
  }

  if (centre === 'dub-ready') {
    if (lip.ready) {
      const url = media?.video.lipsync?.download_url ?? null;
      return {
        kind: 'lipsync-ready',
        status: 'Lip-synced video ready',
        sub: null,
        tone: 'ok',
        action: url ? { kind: 'download', label: 'Download video', href: url } : null,
        dubDownloadUrl: dubUrl,
      };
    }
    return {
      kind: 'dub-ready',
      status: 'Dubbed video ready',
      sub: lip.running ? 'Lip-sync is still rendering.' : "Lip-sync hasn't been run yet.",
      tone: 'ok',
      action: lip.running ? null : { kind: 'sync-lips', label: "Sync speaker's lips" },
      dubDownloadUrl: dubUrl,
    };
  }

  // queued / preparing / source-only — the dub is not usable yet.
  return {
    kind: 'processing',
    status: progress?.state === 'queued' ? 'Queued' : 'Processing…',
    sub: null,
    tone: 'progress',
    action: null,
    dubDownloadUrl: null,
  };
}
