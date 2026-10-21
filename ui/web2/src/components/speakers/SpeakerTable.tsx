import { useStore } from '../../store/useStore';
import { SpeakerRow } from './SpeakerRow';
import { PendingBar } from './PendingBar';
import { EmptyState, Eyebrow, Skeleton } from '../ui/Card';
import { Tooltip } from '../ui/Tooltip';
import { IconWave } from '../ui/Icons';
import { langClass } from '../../lib/lang';

/**
 * The cast: every speaker, their voice, and — where the engine supports it
 * — their delivery.
 *
 * Appears at `milestones.diarization_done`, which is earlier than voices
 * exist. `ui/speakers.py` falls back to the diarization RTTM for ids and
 * speech time, so the cast is on screen as soon as we know who is speaking,
 * with the voice column filling in when the casting stage runs.
 *
 * Ordered by speech time, matching how `assign_voices` already ranks
 * speakers — the protagonist is the row you want at the top.
 *
 * Edits are STAGED. Nothing here calls the pipeline; the bar below shows
 * what re-running costs and an explicit button applies it. Renaming a
 * speaker is the exception and is NOT staged — it changes no audio.
 */
export function SpeakerTable() {
  const table = useStore((s) => s.speakers);
  const staged = useStore((s) => s.staged);
  const stageVoice = useStore((s) => s.stageVoice);
  const stageTone = useStore((s) => s.stageTone);
  const revertSpeaker = useStore((s) => s.revertSpeaker);
  const diarized = useStore((s) => Boolean(s.progress?.milestones?.diarization_done));
  const state = useStore((s) => s.progress?.state);

  if (!diarized) return null;

  if (!table) {
    return (
      <section className="space-y-2">
        <Eyebrow>Cast</Eyebrow>
        <Skeleton className="h-28" />
      </section>
    );
  }

  // A running job's speech synthesis is mid-flight; staging a voice change
  // against it would queue behind a render that is about to overwrite the
  // output it changes.
  const editable = state !== 'running' && state !== 'queued';

  return (
    <section className="space-y-2">
      <div className="panel-head">
        <div className="flex items-baseline gap-2">
          <h2 className="text-[14px] font-semibold text-ink">Cast</h2>
          <span className="text-[12px] text-ink3">
            {table.speakers.length} speaker{table.speakers.length === 1 ? '' : 's'}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11.5px]">
          <span className={langClass(table.language)}>
            <span className="text-ink3">{table.engine_label}</span>
          </span>
          {!table.supports_tone ? (
            <Tooltip label="Only the Bodhan voices accept a delivery style, so the control is not shown for this engine — setting it here would change nothing.">
              <span className="cursor-help rounded-md border border-line px-1.5 py-px text-ink4">
                no delivery control
              </span>
            </Tooltip>
          ) : null}
        </div>
      </div>

      <div className="card divide-y divide-line overflow-hidden">
        {table.speakers.length === 0 ? (
          <EmptyState
            icon={<IconWave width={18} height={18} />}
            title="No speakers were separated out"
            body="Speaker annotation finished without producing any speakers. That usually means the clip has no clear speech, or it stopped before writing its output."
          />
        ) : (
          table.speakers.map((sp) => (
            <SpeakerRow
              key={sp.id}
              speaker={sp}
              table={table}
              staged={staged[sp.id]}
              editable={editable}
              onVoice={(v) => stageVoice(sp.id, v)}
              onTone={(t) => stageTone(sp.id, t)}
              onRevert={() => revertSpeaker(sp.id)}
            />
          ))
        )}
      </div>

      {!editable && table.speakers.length > 0 ? (
        <p className="px-1 text-[12px] text-ink3">
          Voices can be recast once this dub finishes — the speech is being synthesised right
          now. You can still rename speakers.
        </p>
      ) : null}

      <PendingBar />
    </section>
  );
}
