import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../store/useStore';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconSpinner } from '../ui/Icons';
import { StagedText, StagedValue } from '../ui/StagedValue';
import { fmtClock } from '../../lib/format';
import type { AgentScriptRow, ScriptEdit } from '../../api/types';

/**
 * The full cue list, beside the player, with follow-along highlighting.
 *
 * Shows ALL cues in a scrollable container. The cue matching the current video
 * playback time is highlighted; a "Go to current" button scrolls to it. Each
 * cue has a ▶ button to seek the video to its start. The full editing surface
 * (inline text rewrite, mute, speed, staged indicators) that used to live in
 * the separate "Script" section is merged in here.
 *
 * "Current" is the last cue whose `start` has passed, not strictly
 * `start <= t < end`: between two cues there is a silent gap, and holding the
 * just-finished line on screen through that gap reads better than blanking to
 * nothing for half a second.
 */
export function NowPlayingCues({
  segments,
  busy,
  onMute,
  onStageScript,
}: {
  segments: AgentScriptRow[];
  busy: boolean;
  onMute: (segId: string) => void;
  onStageScript: (edits: ScriptEdit[]) => void;
}) {
  const t = useStore((s) => s.videoTime);
  const requestSeek = useStore((s) => s.requestSeek);
  const selectedCue = useStore((s) => s.selectedCue);
  const setSelectedCue = useStore((s) => s.setSelectedCue);

  const listRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<Map<string, HTMLLIElement>>(new Map());

  // Hand-edited lines, keyed by segment id.
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [editing, setEditing] = useState<string | null>(null);
  // Tracks which cues the user has muted this session (toggle visual).
  const [mutedIds, setMutedIds] = useState<Set<string>>(new Set());

  const dirty = Object.keys(drafts);

  // Start-ordered.
  const rows = useMemo(
    () => [...segments].sort((a, b) => a.start - b.start),
    [segments],
  );

  const currentIdx = useMemo(() => {
    let idx = -1;
    for (let i = 0; i < rows.length; i++) {
      if (rows[i].start <= t) idx = i;
      else break;
    }
    return idx;
  }, [rows, t]);

  const currentId = currentIdx >= 0 ? rows[currentIdx].id : null;

  // Scroll to the current cue.
  const scrollToCurrent = useCallback(() => {
    if (!currentId) return;
    const el = rowRefs.current.get(currentId);
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [currentId]);

  // Auto-scroll when current cue changes, but only if the user hasn't
  // manually scrolled away — we detect "near bottom or near the cue" by
  // checking if the highlighted row is already in view.
  const lastAutoScrollId = useRef<string | null>(null);
  useEffect(() => {
    if (!currentId || currentId === lastAutoScrollId.current) return;
    lastAutoScrollId.current = currentId;
    const el = rowRefs.current.get(currentId);
    if (!el || !listRef.current) return;
    const container = listRef.current;
    const ct = container.getBoundingClientRect();
    const et = el.getBoundingClientRect();
    // Only auto-scroll if the row is NOT already visible.
    if (et.top < ct.top || et.bottom > ct.bottom) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentId]);

  const stageEdits = useCallback(() => {
    const edits: ScriptEdit[] = Object.entries(drafts).map(([segment_id, text]) => ({
      segment_id,
      text,
    }));
    if (!edits.length) return;
    onStageScript(edits);
    setDrafts({});
    setEditing(null);
  }, [drafts, onStageScript]);

  return (
    <section className="card flex flex-col p-4">
      <div className="mb-2 flex items-center gap-2">
        <Eyebrow>Cues{rows.length ? ` · ${rows.length}` : ''}</Eyebrow>
        <span className="text-[11px] text-ink4">
          Click text to rewrite · nothing re-renders until apply
        </span>
        <button
          type="button"
          onClick={scrollToCurrent}
          title="Scroll to the cue matching the current playback position"
          className="ml-auto flex items-center gap-1 rounded-lg border border-edge/60 px-2 py-1
                     text-[11px] text-ink3 transition-colors hover:border-edge hover:text-ink"
        >
          <span aria-hidden="true">⏱</span> Go to current
        </button>
      </div>

      {dirty.length ? (
        <div className="mb-2 flex flex-wrap items-center gap-3 rounded-lg bg-amber-400/10 px-3 py-2 ring-1 ring-amber-400/40">
          <span className="text-[12.5px] text-ink">
            {dirty.length} line{dirty.length === 1 ? '' : 's'} rewritten
          </span>
          <span className="text-[11px] text-ink4">
            Replays speech, mix and lip-sync.
          </span>
          <div className="ml-auto flex gap-2">
            <Button
              variant="ghost"
              onClick={() => {
                setDrafts({});
                setEditing(null);
              }}
            >
              Discard
            </Button>
            <Button onClick={stageEdits} disabled={busy}>
              {busy ? <IconSpinner /> : null} Review &amp; apply
            </Button>
          </div>
        </div>
      ) : null}

      {rows.length === 0 ? (
        <p className="text-[12.5px] text-ink4">
          Cues appear here once there is a transcript.
        </p>
      ) : (
        <div
          ref={listRef}
          className="overflow-y-auto"
          style={{ maxHeight: 'calc(100vh - 16rem)' }}
        >
          <ol className="space-y-1">
            {rows.map((r, i) => {
              const isCurrent = i === currentIdx;
              const isSelected = selectedCue === r.id;
              const shown = drafts[r.id] ?? r.staged_text ?? r.text ?? '';
              const was = drafts[r.id] !== undefined
                ? (r.staged_text ?? r.text)
                : r.staged_text ? r.text : null;

              return (
                <li
                  key={r.id}
                  ref={(el) => {
                    if (el) rowRefs.current.set(r.id, el);
                    else rowRefs.current.delete(r.id);
                  }}
                  className={`rounded-lg px-2.5 py-1.5 text-[12.5px] transition-colors ${
                    isSelected
                      ? 'bg-brand/10 ring-2 ring-brand'
                      : isCurrent
                        ? 'bg-brand/10 ring-1 ring-brand/40'
                        : drafts[r.id] !== undefined
                          ? 'bg-amber-400/10 ring-1 ring-amber-400/40'
                          : r.staged
                            ? 'bg-brand/5 ring-1 ring-brand/20'
                            : 'odd:bg-surface/40'
                  }`}
                >
                  <div className="flex items-baseline gap-2">
                    {/* Cue number — click copies "cue N" for the edit agent. */}
                    <button
                      type="button"
                      onClick={() => {
                        // Aim the edit workspace at this cue (toggle off if it
                        // is already the selection), and keep the old
                        // copy-"cue N" convenience for a typed request.
                        setSelectedCue(isSelected ? null : r.id);
                        try {
                          void navigator.clipboard?.writeText(`cue ${r.cue_no}`);
                        } catch {
                          /* Clipboard denial on insecure origins */
                        }
                      }}
                      title={`${r.id} — click to aim edits at cue ${r.cue_no}`}
                      className={`flex-none font-mono text-[11px] rounded px-1 -mx-1 hover:bg-brand/10 ${
                        isSelected ? 'text-brand-ink font-semibold' : 'text-brand hover:text-brand-ink'
                      }`}
                    >
                      #{r.cue_no}
                    </button>
                    <span className="flex-none font-mono text-[11px] text-ink4">
                      {fmtClock(r.start)}
                    </span>
                    {r.speaker ? (
                      <span className="flex-none font-mono text-[11px] text-ink3">{r.speaker}</span>
                    ) : null}

                    {/* Inline text edit or display. */}
                    {editing === r.id ? (
                      <input
                        autoFocus
                        className="min-w-0 flex-1 rounded border border-brand/40 bg-surface px-1.5 py-0.5 text-ink outline-none"
                        defaultValue={shown}
                        onBlur={(e) => {
                          const next = e.currentTarget.value.trim();
                          const before = r.staged_text ?? r.text;
                          setEditing(null);
                          setDrafts((d) => {
                            const copy = { ...d };
                            if (!next || next === before) delete copy[r.id];
                            else copy[r.id] = next;
                            return copy;
                          });
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') e.currentTarget.blur();
                          if (e.key === 'Escape') {
                            e.currentTarget.value = shown;
                            e.currentTarget.blur();
                          }
                        }}
                      />
                    ) : (
                      <button
                        type="button"
                        className="min-w-0 flex-1 cursor-text text-left text-ink hover:underline decoration-dotted underline-offset-2"
                        title="Click to edit this line"
                        onClick={() => setEditing(r.id)}
                      >
                        {shown || <em className="text-ink4">—</em>}
                      </button>
                    )}

                    {/* Speed. */}
                    {r.staged_speed != null && r.staged_speed !== r.speed ? (
                      <span className="ml-auto flex-none font-mono text-[11px]">
                        <StagedValue
                          current={`${r.speed.toFixed(2)}×`}
                          staged={`${r.staged_speed.toFixed(2)}×`}
                          label={`Speed of the cue at ${fmtClock(r.start)}`}
                        />
                      </span>
                    ) : r.speed !== 1 ? (
                      <span
                        className="ml-auto flex-none font-mono text-[11px] text-ink3"
                        title="How fast this cue is spoken"
                      >
                        {r.speed.toFixed(2)}×
                      </span>
                    ) : null}

                    {/* Staged indicator. */}
                    {was !== null ? (
                      <span className="flex-none text-[11px] text-blue-ink">
                        {drafts[r.id] !== undefined ? 'edited' : 'staged'}
                      </span>
                    ) : r.staged ? (
                      <span className="flex-none text-[11px] text-blue-ink">staged</span>
                    ) : null}

                    {/* ▶ Go to video: seek the player to this cue's start. */}
                    <button
                      type="button"
                      onClick={() => requestSeek(r.start)}
                      title={`Jump video to ${fmtClock(r.start)}`}
                      aria-label={`Seek to cue at ${fmtClock(r.start)}`}
                      className="flex-none rounded p-0.5 text-ink4 hover:text-brand
                                 transition-colors"
                    >
                      <span aria-hidden="true" className="text-[11px]">▶</span>
                    </button>

                    {/* Mute cue. */}
                    <button
                      type="button"
                      onClick={() => {
                        setMutedIds((s) => {
                          const next = new Set(s);
                          if (next.has(r.id)) next.delete(r.id);
                          else next.add(r.id);
                          return next;
                        });
                        onMute(r.id);
                      }}
                      disabled={busy}
                      title={mutedIds.has(r.id) ? 'This cue is muted (staged)' : 'Mute this cue — it will be silent in the dub'}
                      aria-label={`${mutedIds.has(r.id) ? 'Unmute' : 'Mute'} cue at ${fmtClock(r.start)}`}
                      className={`flex-none rounded p-1 transition-colors disabled:opacity-30 ${
                        mutedIds.has(r.id) ? 'text-danger' : 'text-ink4 hover:text-danger'
                      }`}
                    >
                      {mutedIds.has(r.id) ? <IconMicOff /> : <IconMic />}
                    </button>
                  </div>

                  {/* Before/after diff, under the line. */}
                  {was !== null && editing !== r.id ? (
                    <StagedText current={was} staged={shown} />
                  ) : null}
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </section>
  );
}

/** microphone.svg — unmuted (default). */
function IconMic() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" role="img" aria-label="Microphone">
  <rect x="9" y="2.5" width="6" height="12" rx="3"/>
  <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0"/>
  <path d="M12 18v3.5"/>
  <path d="M8.5 21.5h7"/>
</svg>
  );
}

/** Microphone-Off.svg — muted. */
function IconMicOff() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="9" y="2.5" width="6" height="12" rx="3"/>
  <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0"/>
  <path d="M12 18v3.5"/>
  <path d="M8.5 21.5h7"/>
  <path d="M3 3l18 18"/>
</svg>
  );
}
