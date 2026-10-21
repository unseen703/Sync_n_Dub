import { useEffect, useRef } from 'react';
import { cn } from '../../lib/cn';
import type { MediaRef } from '../../api/types';

/**
 * Two `<video>` elements stacked in one box, crossfaded on toggle.
 *
 * The point of an Original / Lip-synced toggle is A/B-ing a specific moment
 * — "does her mouth match on *this* word". Swapping one element's `src`
 * loses the position and the play state every time, which makes exactly
 * that comparison impossible. So both are mounted, the hidden one is kept
 * roughly in sync, and switching copies `currentTime` and the paused state
 * across before the 180 ms crossfade.
 *
 * `preload="metadata"` on the inactive track: `/files/` serves HTTP range
 * requests (verified on the Starlette FileResponse), so seeking works, but
 * eagerly buffering a second 45 MB render nobody has asked for does not.
 */
export function DualVideoPlayer({
  original,
  lipsync,
  active,
  poster,
  className,
}: {
  original: MediaRef | null;
  lipsync: MediaRef | null;
  active: 'original' | 'lipsync';
  poster?: string | null;
  className?: string;
}) {
  const aRef = useRef<HTMLVideoElement>(null);
  const bRef = useRef<HTMLVideoElement>(null);
  const prev = useRef<'original' | 'lipsync'>(active);

  useEffect(() => {
    if (prev.current === active) return;
    const from = prev.current === 'original' ? aRef.current : bRef.current;
    const to = active === 'original' ? aRef.current : bRef.current;
    prev.current = active;
    if (!from || !to) return;

    const t = from.currentTime;
    const wasPlaying = !from.paused && !from.ended;
    from.pause();

    const place = () => {
      // readyState < 1 means no metadata yet, and setting currentTime then
      // is silently dropped — which is how a toggle lands back at 0:00.
      if (to.readyState >= 1) {
        try {
          to.currentTime = t;
        } catch {
          /* seek refused while the element is reloading */
        }
        if (wasPlaying) void to.play().catch(() => undefined);
      } else {
        to.addEventListener('loadedmetadata', place, { once: true });
      }
    };
    place();
  }, [active]);

  const common =
    'absolute inset-0 h-full w-full bg-black object-contain transition-opacity duration-[180ms]';

  return (
    <div
      className={cn(
        'relative aspect-video w-full overflow-hidden rounded-2xl border border-line bg-black',
        className,
      )}
    >
      {original ? (
        <video
          ref={aRef}
          src={original.url}
          poster={poster ?? undefined}
          controls
          playsInline
          preload={active === 'original' ? 'auto' : 'metadata'}
          className={cn(common, active === 'original' ? 'z-10 opacity-100' : 'z-0 opacity-0')}
        />
      ) : null}
      {lipsync ? (
        <video
          ref={bRef}
          src={lipsync.url}
          poster={poster ?? undefined}
          controls
          playsInline
          preload={active === 'lipsync' ? 'auto' : 'metadata'}
          className={cn(common, active === 'lipsync' ? 'z-10 opacity-100' : 'z-0 opacity-0')}
        />
      ) : null}
      {!original && !lipsync ? (
        <div className="absolute inset-0 grid place-items-center px-6 text-center text-[12.5px] text-ink3">
          The video for this dub is not on disk yet.
        </div>
      ) : null}
    </div>
  );
}
