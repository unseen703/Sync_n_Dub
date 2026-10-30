import { MotionConfig } from 'framer-motion';
import { TopBar } from './TopBar';
import { Landing } from './Landing';
import { EditPanel } from '../agent/EditPanel';
import { TalkingHeadPage } from '../talkinghead/TalkingHeadPage';
import { ResultHeader } from '../progress/ResultHeader';
import { FailureCard } from '../progress/FailureCard';
import { VideoStage } from '../video/VideoStage';
import { SpeakerTable } from '../speakers/SpeakerTable';
import { Eyebrow, Skeleton } from '../ui/Card';
import { Button } from '../ui/Button';
import { IconAlert } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import { deriveCentreState } from '../../lib/centre';
import { normalizeState } from '../../lib/format';

/** Job workspace. It reads the selected job and renders its media, progress, editing controls, or a landing state. */
export function Stage() {
  const selected = useStore((s) => s.selected);
  const progress = useStore((s) => s.progress);
  const media = useStore((s) => s.media);
  const jobError = useStore((s) => s.jobError);
  const refreshProgress = useStore((s) => s.refreshProgress);

  const talkingHead = useStore((s) => s.talkingHead);
  const centre = deriveCentreState(progress, media);

  const running = normalizeState(progress?.state) === 'running'
    || normalizeState(progress?.state) === 'queued';

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

        <div className="flex flex-col gap-4">
          <ResultHeader />

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

          {running ? (
            <>
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
              <SpeakerTable />
            </>
          ) : (
            <EditPanel jobId={selected} />
          )}

          <div className="h-4" />
        </div>
      </div>
    </MotionConfig>
  );
}
