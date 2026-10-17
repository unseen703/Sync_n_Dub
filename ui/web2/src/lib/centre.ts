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
