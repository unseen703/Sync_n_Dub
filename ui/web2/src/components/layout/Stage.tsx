import { MotionConfig } from 'framer-motion';
import { TopBar } from './TopBar';
import { Landing } from './Landing';
import { EditPanel } from '../agent/EditPanel';
import { TalkingHeadPage } from '../talkinghead/TalkingHeadPage';
import { StageRibbon } from '../progress/StageRibbon';
import { FailureCard } from '../progress/FailureCard';
import { VideoStage } from '../video/VideoStage';
import { SpeakerTable } from '../speakers/SpeakerTable';
import { Eyebrow, Skeleton } from '../ui/Card';
import { Button } from '../ui/Button';
import { IconAlert } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import { deriveCentreState } from '../../lib/centre';
import { normalizeState } from '../../lib/format';

/**
 * The centre column. The video is central, and everything else arranges
 * itself around it.
 *
 * The reveal order is `deriveCentreState`, which checks `failed` first and
 * unconditionally — but the player stays mounted BELOW the failure banner.
 * A job that dies at lip-sync still has a perfectly good original video,
 * and hiding it would make a partial failure look like a total one.
 *
 * There is no runner log on this screen. It was removed on purpose: this is
 * a user-facing product, and `progress.failure` (from `ui/errors.py`) says
 * what went wrong in a sentence. The raw log is still on disk for whoever
 * is debugging a render.
 *
 * Everything that CHANGES the dub is on this page too, under the player, as
 * of 2026-10-21 — the cast, the chat box, the script, and the lip-sync
 * control, in `EditPanel`. They used to be a separate page at
 * `#/job/<id>/edit`, which meant judging a voice from memory: the video was
 * on one screen and the controls that change it on another.
 */
export function Stage() {
  const selected = useStore((s) => s.selected);
  const progress = useStore((s) => s.progress);
  const media = useStore((s) => s.media);
  const jobError = useStore((s) => s.jobError);
  const refreshProgress = useStore((s) => s.refreshProgress);

  const talkingHead = useStore((s) => s.talkingHead);
  const centre = deriveCentreState(progress, media);

  // The agent replays stages 9b..9h over the run's merged ASR+MT JSON, which
  // only exists once the run has got that far. Offering it mid-run would
  // stage edits against a transcript still being written — the rule that
  // used to gate the top bar's "Edit via agent" button, now gating the panel
  // that replaced it.
  const running = normalizeState(progress?.state) === 'running'
    || normalizeState(progress?.state) === 'queued';

  // Checked before the no-selection branch: this page has no job behind it,
  // so it would otherwise fall through to the landing page.
  if (talkingHead) {
    return (
      <MotionConfig reducedMotion="user">
        <div className="min-h-0 flex-1 overflow-y-auto px-4 lg:px-6">
          <TalkingHeadPage />
        </div>
      </MotionConfig>
    );
  }

  if (!selected) {
    return (
      <MotionConfig reducedMotion="user">
        <div className="min-h-0 flex-1 overflow-y-auto px-4 lg:px-6">
          <Landing />
        </div>
      </MotionConfig>
    );
  }

  return (
    <MotionConfig reducedMotion="user">
      <TopBar />

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 lg:px-5 lg:py-4">
        <div className="mx-auto flex max-w-4xl flex-col gap-4">
          <StageRibbon />

          {centre === 'failed' ? (
            <FailureCard
              failure={progress?.failure ?? null}
              cancelled={progress?.state === 'cancelled'}
            />
          ) : null}

          {jobError ? (
            <div className="flex items-center gap-3 rounded-xl border border-line bg-surface/70 px-3.5 py-2.5">
              <IconAlert width={15} height={15} className="shrink-0 text-warn" />
              <p className="min-w-0 flex-1 text-[12.5px] leading-relaxed text-ink2">
                We lost touch with the studio server, so this page may be out of date.
              </p>
              <Button size="sm" onClick={() => void refreshProgress()}>
                Retry
              </Button>
            </div>
          ) : null}

          {/* The player, from `video_ready` onward — including under a
              failure banner, which is the whole point of mounting it
              outside the failed branch. */}
          {progress?.milestones?.video_ready ? (
            <VideoStage />
          ) : centre === 'preparing' || centre === 'creating' ? (
            <section className="space-y-2">
              <Eyebrow>{centre === 'creating' ? 'Queued' : 'Getting the video ready'}</Eyebrow>
              <Skeleton className="aspect-video w-full" />
              <p className="t-sm">
                {centre === 'creating'
                  ? 'Waiting for a free GPU — one dub renders at a time on this machine.'
                  : 'The player appears as soon as the video has been fetched and trimmed.'}
              </p>
            </section>
          ) : null}

          {/* Everything you can do to the dub, directly under the dub.
              While the run is still going there is nothing to edit yet, so
              only the cast appears — it fills in from diarization onward,
              several stages before the agent has a transcript to work on. */}
          {running ? <SpeakerTable /> : <EditPanel jobId={selected} />}

          <div className="h-4" />
        </div>
      </div>
    </MotionConfig>
  );
}
