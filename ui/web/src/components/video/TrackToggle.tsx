import { cn } from '../../lib/cn';
import { Tooltip } from '../ui/Tooltip';
import { fmtDuration } from '../../lib/format';

/**
 * The Lip-synced ⇄ Original toggle above the video.
 *
 * **Lip-synced comes first, and it is the default** wherever one has been
 * rendered — it is the thing the run was for, and the original is the
 * comparison. Until the render exists the first segment is disabled and
 * the player shows the original; see `VideoStage` for how the selection
 * resolves, and `useStore.videoTrack` for why an explicit choice outranks
 * the default.
 *
 * Three states, and the distinction between the last two is the whole
 * design:
 *
 *  - not configured  -> this component is not rendered at all. A control
 *                       that can never become live is noise.
 *  - configured, not rendered yet -> the segment is DISABLED and carries
 *                       its own progress bar and an estimate. It never
 *                       falls back to playing the original under a
 *                       "Lip-synced" label; that would defeat the point of
 *                       having a toggle.
 *  - ready           -> a normal segment.
 *
 * The percentage is the job's overall `pct`, not the render's. Stage 9h
 * shells into another conda env and reports nothing finer — the tooltip
 * says so rather than implying a precision we do not have.
 */
export function TrackToggle({
  value,
  onChange,
  lipsyncReady,
  pct,
  running,
  etaS,
}: {
  value: 'original' | 'lipsync';
  onChange: (v: 'original' | 'lipsync') => void;
  lipsyncReady: boolean;
  pct: number;
  running: boolean;
  etaS?: number | null;
}) {
  const seg =
    'relative flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-[12.5px] font-medium transition-colors';

  const lipLabel = lipsyncReady
    ? 'Lip-synced'
    : running
      ? `Lip-synced · ${pct}%`
      : 'Lip-synced · not rendered yet';

  const disabledNote = running
    ? `The lip-synced version is still being rendered${etaS ? `, about ${fmtDuration(etaS)} left` : ''}. The percentage is this dub's overall progress — the renderer itself does not report a finer one.`
    : 'This dub has no lip-synced version yet. It will become selectable the moment one exists — it never quietly plays the original instead.';

  return (
    <div
      role="radiogroup"
      aria-label="Which video to play"
      className="inline-flex items-center gap-1 rounded-xl border border-line bg-surface/70 p-1"
    >
      {lipsyncReady ? (
        <button
          type="button"
          role="radio"
          aria-checked={value === 'lipsync'}
          onClick={() => onChange('lipsync')}
          className={cn(
            seg,
            value === 'lipsync'
              ? 'bg-brand-fill text-white shadow-sm'
              : 'text-ink3 hover:text-ink',
          )}
        >
          Lip-synced
        </button>
      ) : (
        <Tooltip label={disabledNote} side="bottom">
          <span className={cn(seg, 'cursor-help text-ink4')} aria-disabled="true">
            {lipLabel}
            {running ? (
              <span className="absolute inset-x-3 bottom-1 h-0.5 overflow-hidden rounded-full bg-ink/[0.1]">
                <span
                  className="block h-full rounded-full bg-gradient-to-r from-brand to-aqua transition-[width] duration-500"
                  style={{ width: `${Math.max(2, Math.min(100, pct))}%` }}
                />
              </span>
            ) : null}
          </span>
        </Tooltip>
      )}

      <button
        type="button"
        role="radio"
        aria-checked={value === 'original'}
        onClick={() => onChange('original')}
        className={cn(
          seg,
          value === 'original'
            ? 'bg-ink/[0.12] text-ink shadow-sm'
            : 'text-ink3 hover:text-ink',
        )}
      >
        Original
      </button>
    </div>
  );
}
