import { useMemo } from 'react';
import { useStore } from '../../store/useStore';
import { SpeakerRow } from './SpeakerRow';
import { PendingBar } from './PendingBar';
import { EmptyState, Eyebrow, Skeleton } from '../ui/Card';
import { Tooltip } from '../ui/Tooltip';
import { IconWave } from '../ui/Icons';
import { langClass } from '../../lib/lang';
import type { AgentScriptRow } from '../../api/types';

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
/**
 * `scroll` enables vertical scrolling so the cast does not push the cue panel
 * off the page when there are many speakers.
 */
export function SpeakerTable({ scroll = false, segments = [] }: { scroll?: boolean; segments?: AgentScriptRow[] } = {}) {
  const table = useStore((s) => s.speakers);
  const staged = useStore((s) => s.staged);
  const stageVoice = useStore((s) => s.stageVoice);
  const stageTone = useStore((s) => s.stageTone);
  const stageGender = useStore((s) => s.stageGender);
  const revertSpeaker = useStore((s) => s.revertSpeaker);
  const diarized = useStore((s) => Boolean(s.progress?.milestones?.diarization_done));
  const state = useStore((s) => s.progress?.state);
  // What the AGENT has staged, which is a different path to the same rows.
  // A voice, tone or gender changed with the picker lives in `staged` above
  // and never left the browser; one asked for in the chat box is already in
  // the job's `pending.json`. Both paths show on the row.
  const agent = useStore((s) => s.agentParams);
  const awaitingCast = useStore((s) => Boolean(s.progress?.milestones?.awaiting_cast));
  const cast = useStore((s) => s.cast);
  const applying = useStore((s) => s.applying);

  const cuesBySpeaker = useMemo(() => {
    const m = new Map<string, AgentScriptRow[]>();
    for (const seg of [...segments].sort((a, b) => a.start - b.start)) {
      if (!seg.speaker) continue;
      const arr = m.get(seg.speaker);
      if (arr) arr.push(seg);
      else m.set(seg.speaker, [seg]);
    }
    return m;
  }, [segments]);

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

      <div
        className={`card divide-y divide-line overflow-hidden ${
          scroll ? 'max-h-64 overflow-y-auto scroll-smooth' : ''
        }`}
      >
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
              fromAgent={agent?.speakers.find((a) => a.id === sp.id) ?? null}
              cues={cuesBySpeaker.get(sp.id)}
              editable={editable}
              onVoice={(v) => stageVoice(sp.id, v)}
              onTone={(t) => stageTone(sp.id, t)}
              onGender={(g) => stageGender(sp.id, g)}
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

      {awaitingCast ? (
        <div className="card flex flex-wrap items-center justify-between gap-3
                        border-brand/40 bg-brand/[0.06] px-3.5 py-3">
          <p className="max-w-[32rem] text-[12.5px] text-ink3">
            Set a gender and voice for each speaker, or leave them on{' '}
            <span className="font-medium text-ink">Auto-detect</span>. Casting
            runs the speech and mix with your choices — lip-sync is a separate
            step afterwards.
          </p>
          <button
            type="button"
            onClick={() => void cast()}
            disabled={applying || !editable}
            className="shrink-0 rounded-lg bg-brand px-3.5 py-2 text-[13px] font-semibold
                       text-white transition-colors hover:bg-brand/90 disabled:opacity-60"
          >
            {applying ? 'Casting…' : 'Cast & continue'}
          </button>
        </div>
      ) : (
        <PendingBar />
      )}
    </section>
  );
}
