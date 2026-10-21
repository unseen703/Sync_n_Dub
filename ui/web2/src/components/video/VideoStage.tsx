import { useStore } from '../../store/useStore';
import { DualVideoPlayer } from './DualVideoPlayer';
import { TrackToggle } from './TrackToggle';
import { AudioBar } from './AudioBar';
import { Skeleton } from '../ui/Card';
import { IconDownload } from '../ui/Icons';
import { lipsyncToggleState } from '../../lib/centre';
import { fmtBytes } from '../../lib/format';

/**
 * The centre column's media block: toggle, player, then the two audio
 * tracks directly beneath it.
 *
 * Mounted from `video_ready`, which lands several stages before diarization
 * — the cut clip exists long before we know who is speaking, so holding the
 * player back until diarization would be withholding something we already
 * have.
 */
export function VideoStage() {
  const media = useStore((s) => s.media);
  const progress = useStore((s) => s.progress);
  const track = useStore((s) => s.videoTrack);
  const setTrack = useStore((s) => s.setVideoTrack);

  const lip = lipsyncToggleState(progress, media);

  // Which track actually plays, in two steps.
  //
  //  1. `track === null` means nobody has chosen, so follow the DEFAULT:
  //     lip-synced wherever one has been rendered, original until then.
  //  2. Whatever came out of step 1, a lip-synced track that is not ready
  //     falls back to the original — a re-render can take it away again
  //     mid-session, and a dead player is worse than the source video.
  //
  // Step 2 is a derivation, deliberately NOT a `setTrack('original')` in an
  // effect the way it used to be. Writing the fallback back into the store
  // destroyed the user's choice: once a re-render had bounced them to the
  // original, the finished render could not return them to the lip-synced
  // track, because the store now said they had picked the original.
  const wanted = track ?? (lip.ready ? 'lipsync' : 'original');
  const active = wanted === 'lipsync' && !lip.ready ? 'original' : wanted;

  if (!media) return <Skeleton className="aspect-video w-full" />;

  const current = active === 'lipsync' ? media.video.lipsync : media.video.original;

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        {lip.configured ? (
          <TrackToggle
            value={active}
            onChange={setTrack}
            lipsyncReady={lip.ready}
            pct={lip.pct}
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

      <DualVideoPlayer
        original={media.video.original}
        lipsync={media.video.lipsync}
        active={active}
        poster={media.poster_url}
        className="mx-auto max-h-[56vh] max-w-4xl"
      />

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
