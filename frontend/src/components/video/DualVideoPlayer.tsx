import { useEffect, useRef } from 'react';
import { cn } from '../../lib/cn';
import type { MediaRef, VideoTrack } from '../../api/types';

/** Multi-track video player. Inputs are media tracks, the active track, and optional seek events; output is the active player. */
export function DualVideoPlayer({
  tracks,
  active,
  poster,
  className,
  onTime,
  seekTo,
}: {
  tracks: Partial<Record<VideoTrack, MediaRef | null>>;
  active: VideoTrack;
  poster?: string | null;
  className?: string;
  onTime?: (t: number) => void;
  seekTo?: { t: number; nonce: number } | null;
}) {
  const els = useRef<Partial<Record<VideoTrack, HTMLVideoElement | null>>>({});
  const prev = useRef<VideoTrack>(active);

  useEffect(() => {
    if (!seekTo) return;
    const el = els.current[active];
    if (!el) return;
    const jump = () => {
      try {
        el.currentTime = seekTo.t;
      } catch {}
      void el.play().catch(() => undefined);
    };
    if (el.readyState >= 1) jump();
    else el.addEventListener('loadedmetadata', jump, { once: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seekTo?.nonce]);

  useEffect(() => {
    if (prev.current === active) return;
    const from = els.current[prev.current];
    const to = els.current[active];
    prev.current = active;
    if (!from || !to) return;

    const t = from.currentTime;
    const wasPlaying = !from.paused && !from.ended;
    from.pause();

    const place = () => {

      if (to.readyState >= 1) {
        try {
          to.currentTime = t;
        } catch {}
        if (wasPlaying) void to.play().catch(() => undefined);
      } else {
        to.addEventListener('loadedmetadata', place, { once: true });
      }
    };
    place();
  }, [active]);

  const common =
    'absolute inset-0 h-full w-full bg-black object-contain transition-opacity duration-[180ms]';

  const mounted = (Object.keys(tracks) as VideoTrack[]).filter((k) => tracks[k]);

  return (
    <div
      className={cn(
        'relative aspect-video w-full overflow-hidden rounded-2xl border border-line bg-black',
        className,
      )}
    >
      {mounted.map((key) => (
        <video
          key={key}
          ref={(el) => {
            els.current[key] = el;
          }}
          src={tracks[key]!.url}
          poster={poster ?? undefined}
          controls
          playsInline
          onTimeUpdate={
            active === key ? (e) => onTime?.(e.currentTarget.currentTime) : undefined
          }
          preload={active === key ? 'auto' : 'metadata'}
          className={cn(common, active === key ? 'z-10 opacity-100' : 'z-0 opacity-0')}
        />
      ))}
      {!mounted.length ? (
        <div className="absolute inset-0 grid place-items-center px-6 text-center text-[12.5px] text-ink3">
          The video for this dub is not on disk yet.
        </div>
      ) : null}
    </div>
  );
}
