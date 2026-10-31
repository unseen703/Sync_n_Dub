import { useCallback } from 'react';
import { useStore } from '../../store/useStore';
import { DualVideoPlayer } from './DualVideoPlayer';
import { TrackToggle } from './TrackToggle';
import { AudioBar } from './AudioBar';
import { Skeleton } from '../frontend/Card';
import { IconDownload } from '../frontend/Icons';
import { lipsyncToggleState } from '../../lib/centre';
import { fmtBytes } from '../../lib/format';
import type { VideoTrack } from '../../api/types';

/** Media stage. It reads available job media and renders the selected video track with its audio controls. */
export function VideoStage() {
  const setVideoPlayerHeight = useStore((s) => s.setVideoPlayerHeight);
  const playerRef = useCallback(
    (el: HTMLDivElement | null) => {
      if (!el || typeof ResizeObserver === 'undefined') {
        setVideoPlayerHeight(null);
        return;
      }
      const ro = new ResizeObserver(([entry]) => {
        setVideoPlayerHeight(entry.contentRect.height);
      });
      ro.observe(el);
      setVideoPlayerHeight(el.getBoundingClientRect().height);
      (el as HTMLDivElement & { __ro?: ResizeObserver }).__ro = ro;
    },
    [setVideoPlayerHeight],
  );
  const media = useStore((s) => s.media);
  const progress = useStore((s) => s.progress);
  const track = useStore((s) => s.videoTrack);
  const setTrack = useStore((s) => s.setVideoTrack);
  const setVideoTime = useStore((s) => s.setVideoTime);
  const videoSeek = useStore((s) => s.videoSeek);

  const lip = lipsyncToggleState(progress, media);

  const enhancedReady = Boolean(media?.video.enhanced);

  const best: VideoTrack = enhancedReady ? 'enhanced' : lip.ready ? 'lipsync' : 'original';
  const wanted = track ?? best;
  const available: Record<VideoTrack, boolean> = {
    enhanced: enhancedReady,
    lipsync: lip.ready,
    original: true,
  };
  const active: VideoTrack = available[wanted] ? wanted : best;

  if (!media) return <Skeleton className="aspect-video w-full" />;

  const current = media.video[active];

  const warning =
    active === 'enhanced'
      ? media.video.enhanced_warning
      : active === 'lipsync'
        ? media.video.lipsync_warning
        : null;

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        {lip.configured || enhancedReady ? (
          <TrackToggle
            value={active}
            onChange={setTrack}
            lipsyncReady={lip.ready}
            enhancedReady={enhancedReady}
            running={lip.running || progress?.state === 'running'}
          />
        ) : (
          <div className="eyebrow">Source video</div>
        )}

        <div className="flex items-center gap-2">
          {media.video.dubbed_no_lipsync ? (
            <a
              href={media.video.dubbed_no_lipsync.download_url}
              download
              className="inline-flex items-center gap-1.5 rounded-lg border border-edge/60 px-2.5 py-1.5
                         text-[12px] text-ink2 transition-colors hover:border-edge hover:text-ink"
            >
              <IconDownload width={13} height={13} />
              Dubbed video
              <span className="font-mono text-[11px] text-ink4">
                {fmtBytes(media.video.dubbed_no_lipsync.bytes)}
              </span>
            </a>
          ) : null}
          {current ? (
            <a
              href={current.download_url}
              download
              className="inline-flex items-center gap-1.5 rounded-lg border border-edge/60 px-2.5 py-1.5
                         text-[12px] text-ink2 transition-colors hover:border-edge hover:text-ink"
            >
              <IconDownload width={13} height={13} />
              {current.label}
              <span className="font-mono text-[11px] text-ink4">{fmtBytes(current.bytes)}</span>
            </a>
          ) : null}
        </div>
      </div>

      {warning ? (
        <div className="mx-auto max-w-4xl rounded-lg border border-amber-400/40 bg-amber-50 px-3 py-2
                        text-[13px] text-amber-800 dark:border-amber-500/30 dark:bg-amber-950/40 dark:text-amber-300">
          {warning}
        </div>
      ) : null}

      <div ref={playerRef} className="mx-auto max-w-4xl">
        <DualVideoPlayer
          tracks={{
            enhanced: media.video.enhanced,
            lipsync: media.video.lipsync,
            original: media.video.original,
          }}
          active={active}
          poster={media.poster_url}
          onTime={setVideoTime}
          seekTo={videoSeek}
          className="max-h-[56vh] w-full"
        />
      </div>

      <div className="mx-auto max-w-4xl">
        <AudioBar
          original={media.audio.original}
          dubbed={media.audio.dubbed}
          note={media.audio.original?.full_quality === false ? media.audio.original.note : null}
        />
      </div>
    </section>
  );
}
