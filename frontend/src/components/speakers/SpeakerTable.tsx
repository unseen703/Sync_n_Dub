import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../store/useStore';
import { SpeakerRow } from './SpeakerRow';
import { PendingBar } from './PendingBar';
import { EmptyState, Eyebrow, Skeleton } from '../frontend/Card';
import { Tooltip } from '../frontend/Tooltip';
import { IconWave } from '../frontend/Icons';
import type { AgentScriptRow } from '../../api/types';

/** Cast editor. Inputs control scrolling and script rows; output is one editable row per speaker. */
export function SpeakerTable({ scroll = false, segments = [] }: { scroll?: boolean; segments?: AgentScriptRow[] } = {}) {
  const table = useStore((s) => s.speakers);
  const staged = useStore((s) => s.staged);
  const stageVoice = useStore((s) => s.stageVoice);
  const stageTone = useStore((s) => s.stageTone);
  const stageGender = useStore((s) => s.stageGender);
  const revertSpeaker = useStore((s) => s.revertSpeaker);
  const diarized = useStore((s) => Boolean(s.progress?.milestones?.diarization_done));
  const state = useStore((s) => s.progress?.state);

  const agent = useStore((s) => s.agentParams);
  const awaitingCast = useStore((s) => Boolean(s.progress?.milestones?.awaiting_cast));
  const selected = useStore((s) => s.selected);

  const [expandedId, setExpandedId] = useState<string | null>(null);

  const scrolled = useRef<string | null>(null);
  useEffect(() => {
    if (!awaitingCast || !selected || scrolled.current === selected) return;
    scrolled.current = selected;
    requestAnimationFrame(() => {
      document.getElementById('cast-panel')
        ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }, [awaitingCast, selected]);

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

  // Auto-expand when there's only one speaker
  useEffect(() => {
    if (table && table.speakers.length === 1 && expandedId === null) {
      setExpandedId(table.speakers[0].id);
    }
  }, [table, expandedId]);

  if (!diarized) return null;

  if (!table) {
    return (
      <section className="space-y-2">
        <Eyebrow>Cast</Eyebrow>
        <Skeleton className="h-28" />
      </section>
    );
  }

  const editable = state !== 'running' && state !== 'queued';

  return (
    <section id="cast-panel" className="space-y-2">
      <div className="panel-head">
        <div className="flex items-baseline gap-2">
          <h2 className="text-[15px] font-semibold text-ink">
            Cast &middot; {table.speakers.length} speaker{table.speakers.length === 1 ? '' : 's'}
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[12.5px] text-ink3">
            {table.speakers.length} detected
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
          scroll ? 'max-h-[32rem] overflow-y-auto scroll-smooth' : ''
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
              expanded={expandedId === sp.id}
              onToggle={() => setExpandedId(expandedId === sp.id ? null : sp.id)}
              onVoice={(v) => stageVoice(sp.id, v)}
              onTone={(t) => stageTone(sp.id, t)}
              onGender={(g) => stageGender(sp.id, g)}
              onRevert={() => revertSpeaker(sp.id)}
            />
          ))
        )}
      </div>

      {!editable && table.speakers.length > 0 ? (
        <p className="px-1 text-[13px] text-ink3">
          Voices can be recast once this dub finishes — the speech is being synthesised right
          now. You can still rename speakers.
        </p>
      ) : null}

      {awaitingCast ? (
        <p className="px-1 text-[13px] text-ink3">
          Pick a voice for every speaker, then render dubbed audio from the{' '}
          <span className="font-medium text-ink">Next steps</span> card below.
        </p>
      ) : (
        <PendingBar />
      )}
    </section>
  );
}
