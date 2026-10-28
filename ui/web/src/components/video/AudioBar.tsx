import { useEffect, useRef, useState } from 'react';
import * as audio from '../../lib/audio';
import { useAudioPreview } from '../../hooks/useAudioPreview';
import { cn } from '../../lib/cn';
import { fmtBytes, fmtClock } from '../../lib/format';
import { IconAlert, IconDownload, IconPause, IconPlay } from '../ui/Icons';
import { Tooltip } from '../ui/Tooltip';
import type { MediaRef } from '../../api/types';

/**
 * Requirement 9: play and download for BOTH original and dubbed audio,
 * directly below the video.
 *
 * Both share one `HTMLAudioElement`, so starting one stops the other.
 * That is the behaviour you want here: the comparison is sequential, and
 * two tracks of the same lecture playing at once is just noise.
 *
 * The `full_quality: false` note is NOT optional chrome. When stems were
 * skipped there is no 44.1 kHz stereo extraction on disk and the only
 * "original audio" is the 16 kHz mono file made for diarization. Playing
 * that unlabelled next to a 48 kHz dub makes the dub sound better than it
 * is, and would be the UI lying about what the pipeline produced.
 */
function Track({ track, tone }: { track: MediaRef | null; tone: 'slate' | 'accent' }) {
  const playingSrc = useAudioPreview();
  const isPlaying = Boolean(track && playingSrc === track.url);
  const [frac, setFrac] = useState(0);
  const [dur, setDur] = useState<number | null>(null);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (!isPlaying) {
      if (raf.current) cancelAnimationFrame(raf.current);
      return undefined;
    }
    const loop = () => {
      const el = audio.element();
      if (el && isFinite(el.duration) && el.duration > 0) {
        setFrac(el.currentTime / el.duration);
        setDur(el.duration);
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [isPlaying]);

  if (!track) {
    return (
      <div className="flex min-w-0 flex-1 items-center gap-2.5 rounded-xl border border-dashed border-edge/40 px-3 py-2.5">
        <span className="text-[12.5px] text-ink4">
          {tone === 'accent'
            ? 'The dubbed soundtrack appears here once it is mixed.'
            : 'The original audio appears here once it is extracted.'}
        </span>
      </div>
    );
  }

  const degraded = track.full_quality === false;

  return (
    <div
      className={cn(
        'flex min-w-0 flex-1 items-center gap-2.5 rounded-xl border px-3 py-2 transition-colors',
        tone === 'accent'
          ? 'border-brand-ink/35 bg-brand/[0.12]'
          : 'border-line bg-surface/70',
      )}
    >
      <button
        type="button"
        onClick={() => audio.toggle(track.url)}
        aria-label={`${isPlaying ? 'Pause' : 'Play'} ${track.label}`}
        className={cn(
          'grid h-7 w-7 shrink-0 place-items-center rounded-full transition-colors',
          tone === 'accent'
            ? 'bg-brand-fill text-white hover:bg-brand-soft'
            : 'bg-ink/[0.1] text-ink hover:bg-ink/[0.18]',
        )}
      >
        {isPlaying ? <IconPause width={12} height={12} /> : <IconPlay width={12} height={12} />}
      </button>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-[12.5px] font-medium text-ink">{track.label}</span>
          {degraded ? (
            <Tooltip label={track.note ?? 'Not the source audio.'} side="top">
              <span className="inline-flex cursor-help items-center gap-1 rounded border border-warn/45 bg-warn/10 px-1.5 py-px text-[11px] font-semibold text-warn">
                <IconAlert width={10} height={10} />
                16 kHz mono
              </span>
            </Tooltip>
          ) : null}
        </div>

        {/* Scrubber — only meaningful while this track is the one playing. */}
        <div
          role="presentation"
          onClick={(e) => {
            if (!isPlaying) return;
            const r = e.currentTarget.getBoundingClientRect();
            audio.seek(track.url, (e.clientX - r.left) / r.width);
          }}
          className={cn(
            'mt-1.5 h-1 w-full overflow-hidden rounded-full bg-ink/[0.1]',
            isPlaying && 'cursor-pointer',
          )}
        >
          <div
            className={cn(
              'h-full rounded-full',
              tone === 'accent' ? 'bg-aqua' : 'bg-ink3',
            )}
            style={{ width: isPlaying ? `${Math.round(frac * 100)}%` : '0%' }}
          />
        </div>
      </div>

      <span className="shrink-0 font-mono text-[11.5px] tabular-nums text-ink3">
        {isPlaying && dur ? fmtClock(dur * frac) : fmtBytes(track.bytes)}
      </span>

      <a
        href={track.download_url}
        download
        aria-label={`Download ${track.label}`}
        title={`Download ${track.name}`}
        className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-ink3
                   transition-colors hover:bg-ink/[0.08] hover:text-ink"
      >
        <IconDownload width={14} height={14} />
      </a>
    </div>
  );
}

export function AudioBar({
  original,
  dubbed,
  note,
}: {
  original: MediaRef | null;
  dubbed: MediaRef | null;
  note?: string | null;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex flex-col gap-2 sm:flex-row">
        <Track track={original} tone="slate" />
        <Track track={dubbed} tone="accent" />
      </div>
      {note ? (
        <p className="flex gap-1.5 px-1 text-[11.5px] leading-relaxed text-warn">
          <IconAlert width={12} height={12} className="mt-0.5 shrink-0" />
          {note}
        </p>
      ) : null}
    </div>
  );
}
