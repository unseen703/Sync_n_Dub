import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../store/useStore';
import { SpeakerRow } from './SpeakerRow';
import { PendingBar } from './PendingBar';
import { CastConfirmDialog } from './CastConfirmDialog';
import { EmptyState, Eyebrow, Skeleton } from '../ui/Card';
import { Tooltip } from '../ui/Tooltip';
import { IconWave } from '../ui/Icons';
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
  const applying = useStore((s) => s.applying);
  const selected = useStore((s) => s.selected);
  const [confirmOpen, setConfirmOpen] = useState(false);

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

  const unset = table.speakers.filter((sp) => {
    const st = staged[sp.id];
    return !(st?.voice !== undefined ? st.voice : sp.voice);
  });
  const chosen = table.speakers.length - unset.length;

  return (
    <section id="cast-panel" className="space-y-2">
      <div className="panel-head">
        <div className="flex items-baseline gap-2">
          <h2 className="text-[14px] font-semibold text-ink">Cast</h2>
          <span className="text-[12px] text-ink3">
            {table.speakers.length} speaker{table.speakers.length === 1 ? '' : 's'}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11.5px]">
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
          scroll ? 'max-h-96 overflow-y-auto scroll-smooth' : ''
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
        <div className="relative mt-1 rounded-xl border-[1.5px] border-dashed border-brand/50
                        bg-brand/[0.06] px-4 py-3.5">
          <span
            aria-hidden="true"
            className="absolute -top-[7px] left-7 h-3.5 w-3.5 rotate-45 rounded-[2px]
                       border-l-[1.5px] border-t-[1.5px] border-dashed border-brand/50
                       bg-brand/[0.06]"
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="max-w-[32rem] text-[12.5px] text-ink3">
              Translation complete. <span className="font-medium text-ink">Pick a
              voice for every speaker to carry on</span>
            </p>
            <div className="flex shrink-0 items-center gap-2.5">
              <span className="text-[11.5px] tabular-nums text-ink3">
                {chosen} of {table.speakers.length} voices chosen
              </span>
              <button
                type="button"
                onClick={() => setConfirmOpen(true)}
                disabled={applying || !editable || unset.length > 0}
                title={unset.length
                  ? `${unset.length} speaker${unset.length === 1 ? '' : 's'} still need a voice: ${unset.map((sp) => sp.label || sp.id).join(', ')}`
                  : undefined}
                className="shrink-0 rounded-lg bg-brand px-3.5 py-2 text-[13px] font-semibold
                           text-white transition-colors hover:bg-brand/90 disabled:opacity-60"
              >
                {applying ? 'Casting…' : 'Cast & continue →'}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <PendingBar />
      )}

      <CastConfirmDialog open={confirmOpen} onOpenChange={setConfirmOpen} />
    </section>
  );
}
