/** Derives the main job-panel state. Inputs are progress and media payloads; outputs are display-ready status values. */
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

  return { configured, ready, pct: progress?.pct ?? 0, running };
}

export type OutcomeKind =
  | 'processing'
  | 'dub-ready'
  | 'lipsync-ready'
  | 'enhanced-ready'
  | 'failed';
export type OutcomeTone = 'ok' | 'progress' | 'bad';

export interface OutcomeAction {
  kind: 'sync-lips' | 'download';
  label: string;
  href?: string | null;
}

export interface Outcome {
  kind: OutcomeKind;

  status: string;

  sub: string | null;

  tone: OutcomeTone;
  action: OutcomeAction | null;

  dubDownloadUrl: string | null;
}

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

    const enh = media?.video.enhanced ?? null;
    if (lip.ready && enh && !media?.video.enhanced_warning) {
      return {
        kind: 'enhanced-ready',
        status: 'Enhanced video ready',
        sub: 'Face restoration has been applied to the lip-synced render.',
        tone: 'ok',
        action: { kind: 'download', label: 'Download video', href: enh.download_url },
        dubDownloadUrl: dubUrl,
      };
    }
    if (lip.ready) {
      const url = media?.video.lipsync?.download_url ?? null;
      return {
        kind: 'lipsync-ready',
        status: 'Lip-synced video ready',
        sub: enh ? 'The enhanced version has no audio — its remux failed.' : null,
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

  return {
    kind: 'processing',
    status: progress?.state === 'queued' ? 'Queued' : 'Processing…',
    sub: null,
    tone: 'progress',
    action: null,
    dubDownloadUrl: null,
  };
}
