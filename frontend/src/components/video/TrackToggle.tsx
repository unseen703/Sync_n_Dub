import { cn } from '../../lib/cn';
import { Tooltip } from '../frontend/Tooltip';
import { fmtDuration } from '../../lib/format';
import type { VideoTrack } from '../../api/types';

/** Video-track picker. Inputs describe available tracks and progress; output is the selected track control. */
export function TrackToggle({
  value,
  onChange,
  lipsyncReady,
  enhancedReady,
  running,
  etaS,
}: {
  value: VideoTrack;
  onChange: (v: VideoTrack) => void;
  lipsyncReady: boolean;
  enhancedReady: boolean;
  running: boolean;
  etaS?: number | null;
}) {
  const seg =
    'relative flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-[12.5px] font-medium transition-colors';

  const lipLabel = lipsyncReady
    ? 'Lip-synced'
    : running
      ? 'Lip-synced · rendering'
      : 'Lip-synced · not rendered yet';

  const disabledNote = running
    ? `The lip-synced version is still being rendered${etaS ? `, about ${fmtDuration(etaS)} left` : ''}.`
    : 'This dub has no lip-synced version yet. It will become selectable the moment one exists — it never quietly plays the original instead.';

  return (
    <div
      role="radiogroup"
      aria-label="Which video to play"
      className="inline-flex items-center gap-1 rounded-xl border border-line bg-surface/70 p-1"
    >
      {enhancedReady ? (
        <button
          type="button"
          role="radio"
          aria-checked={value === 'enhanced'}
          onClick={() => onChange('enhanced')}
          className={cn(
            seg,
            value === 'enhanced'
              ? 'bg-brand-fill text-white shadow-sm'
              : 'text-ink3 hover:text-ink',
          )}
        >
          Enhanced
        </button>
      ) : null}

      {lipsyncReady ? (
        <button
          type="button"
          role="radio"
          aria-checked={value === 'lipsync'}
          onClick={() => onChange('lipsync')}
          className={cn(
            seg,
            value === 'lipsync'
              ?

                enhancedReady
                ? 'bg-ink/[0.12] text-ink shadow-sm'
                : 'bg-brand-fill text-white shadow-sm'
              : 'text-ink3 hover:text-ink',
          )}
        >
          Lip-synced
        </button>
      ) : (
        <Tooltip label={disabledNote} side="bottom">
          <span className={cn(seg, 'cursor-help text-ink4')} aria-disabled="true">
            {lipLabel}
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
