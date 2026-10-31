import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../store/useStore';
import { Button } from '../frontend/Button';
import { Eyebrow } from '../frontend/Card';
import { IconSpinner } from '../frontend/Icons';
import { StagedText, StagedValue } from '../frontend/StagedValue';
import { fmtClock } from '../../lib/format';
import { describeLang } from '../../lib/lang';
import type { AgentScriptRow, ScriptEdit } from '../../api/types';

/** Transcript cue list. Inputs are cue data and playback state; output is an editable, seekable cue list. */
export function NowPlayingCues({
  segments,
  busy,
  mutedIds,
  onMute,
  onStageScript,
}: {
  segments: AgentScriptRow[];
  busy: boolean;
  mutedIds: Set<string>;
  onMute: (segId: string, on: boolean) => void;
  onStageScript: (edits: ScriptEdit[]) => void;
}) {
  const matchHeight = useStore((s) => s.videoPlayerHeight);
  const t = useStore((s) => s.videoTime);
  const requestSeek = useStore((s) => s.requestSeek);
  const selectedCue = useStore((s) => s.selectedCue);
  const setSelectedCue = useStore((s) => s.setSelectedCue);
  const rawConfig = useStore((s) => s.spec?.raw_config) as
    | Record<string, unknown>
    | undefined;
  const sourceLangCode =
    (rawConfig?.source_language as string | undefined)
    ?? (rawConfig?.asr_language as string | undefined)
    ?? null;
  // Just the English name, no native-script parenthetical — the button label
  // stays short.
  const sourceLangName =
    sourceLangCode
      ? describeLang(sourceLangCode).replace(/\s*\(.*\)\s*$/, '')
      : 'original';

  const listRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<Map<string, HTMLLIElement>>(new Map());

  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [editing, setEditing] = useState<string | null>(null);
  const [showEnglish, setShowEnglish] = useState(false);

  // Mirror drafts so stageEdits never reads a stale closure — the pending
  // onBlur fires in the same tick as the "Review & apply" click and its
  // setDrafts(...) only lands on the NEXT render, which is why hand edits
  // appeared to disappear on the first click.
  const draftsRef = useRef(drafts);
  useEffect(() => {
    draftsRef.current = drafts;
  }, [drafts]);

  const activeInputRef = useRef<HTMLInputElement | null>(null);

  const dirty = Object.keys(drafts);

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

  const scrollToCurrent = useCallback(() => {
    if (!currentId) return;
    const el = rowRefs.current.get(currentId);
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [currentId]);

  const lastAutoScrollId = useRef<string | null>(null);
  useEffect(() => {
    if (!currentId || currentId === lastAutoScrollId.current) return;
    lastAutoScrollId.current = currentId;
    const el = rowRefs.current.get(currentId);
    if (!el || !listRef.current) return;
    const container = listRef.current;
    const ct = container.getBoundingClientRect();
    const et = el.getBoundingClientRect();

    if (et.top < ct.top || et.bottom > ct.bottom) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentId]);

  const stageEdits = useCallback(() => {
    // Commit any input the user is still typing in — onBlur has not fired yet
    // because the click lands on the button directly.
    activeInputRef.current?.blur();
    const current = draftsRef.current;
    const edits: ScriptEdit[] = Object.entries(current).map(([segment_id, text]) => ({
      segment_id,
      text,
    }));
    if (!edits.length) return;
    onStageScript(edits);
    setDrafts({});
    setEditing(null);
  }, [onStageScript]);

  return (
    <section
      className="card flex flex-col overflow-hidden p-4"
      style={matchHeight ? { height: matchHeight } : undefined}
    >
      <div className="mb-2 flex items-center gap-2">
        <Eyebrow>Script{rows.length ? ` · ${rows.length} cues` : ''}</Eyebrow>
        <span className="text-[12px] text-ink4">
          Click text to rewrite
        </span>
        <button
          type="button"
          onClick={() => setShowEnglish((v) => !v)}
          title={
            showEnglish
              ? `Hide the ${sourceLangName} line under each cue`
              : `See the ${sourceLangName} line under each cue (read-only)`
          }
          aria-pressed={showEnglish}
          className={`ml-auto flex items-center gap-1 rounded-lg border px-2 py-1 text-[12px] transition-colors ${
            showEnglish
              ? 'border-brand/50 text-brand-ink'
              : 'border-edge/60 text-ink3 hover:border-edge hover:text-ink'
          }`}
        >
          {showEnglish ? `Hide ${sourceLangName}` : `See ${sourceLangName}`}
        </button>
        <button
          type="button"
          onClick={scrollToCurrent}
          title="Scroll to the cue matching the current playback position"
          className="flex items-center gap-1 rounded-lg border border-edge/60 px-2 py-1
                     text-[12px] text-ink3 transition-colors hover:border-edge hover:text-ink"
        >
          <span aria-hidden="true">⏱</span> Go to current
        </button>
      </div>

      {dirty.length ? (
        <div className="mb-2 flex flex-wrap items-center gap-3 rounded-lg bg-amber-400/10 px-3 py-2 ring-1 ring-amber-400/40">
          <span className="text-[13px] text-ink">
            {dirty.length} line{dirty.length === 1 ? '' : 's'} rewritten
          </span>
          <span className="text-[12px] text-ink4">
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
        <p className="text-[13px] text-ink4">
          Cues appear here once there is a transcript.
        </p>
      ) : (
        <div
          ref={listRef}
          className="min-h-0 flex-1 overflow-y-auto"
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
                  onClick={() => requestSeek(r.start)}
                  title={`Jump video to ${fmtClock(r.start)}`}
                  className={`cursor-pointer rounded-lg border-l-2 px-3 py-2 text-[13px] transition-colors ${
                    isSelected
                      ? 'border-brand bg-brand/10 ring-2 ring-brand'
                      : isCurrent
                        ? 'border-brand bg-brand/10 ring-1 ring-brand/40'
                        : drafts[r.id] !== undefined
                          ? 'border-transparent bg-amber-400/10 ring-1 ring-amber-400/40'
                          : r.staged
                            ? 'border-transparent bg-brand/5 ring-1 ring-brand/20'
                            : 'border-transparent odd:bg-surface/40'
                  }`}
                >
                  <div className="flex items-baseline gap-2">

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedCue(isSelected ? null : r.id);
                        try {
                          void navigator.clipboard?.writeText(`cue ${r.cue_no}`);
                        } catch {}
                      }}
                      title={`${r.id} — click to aim edits at cue ${r.cue_no}`}
                      className={`flex-none font-mono text-[12px] rounded px-1 -mx-1 hover:bg-brand/10 ${
                        isSelected ? 'text-brand-ink font-semibold' : 'text-brand hover:text-brand-ink'
                      }`}
                    >
                      #{r.cue_no}
                    </button>
                    <span className="flex-none font-mono text-[12px] text-ink4">
                      {fmtClock(r.start)}
                    </span>
                    {r.speaker ? (
                      <span className="flex-none rounded-md bg-surface px-1.5 py-0.5 font-mono text-[12px] text-ink3">
                        {r.speaker}
                      </span>
                    ) : null}

                    {editing === r.id ? (
                      <input
                        autoFocus
                        ref={(el) => {
                          if (el) activeInputRef.current = el;
                          else if (activeInputRef.current && activeInputRef.current !== el)
                            activeInputRef.current = null;
                        }}
                        className="min-w-0 flex-1 rounded border border-brand/40 bg-surface px-1.5 py-0.5 text-ink outline-none"
                        defaultValue={shown}
                        onClick={(e) => e.stopPropagation()}
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
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditing(r.id);
                        }}
                      >
                        {shown || <em className="text-ink4">—</em>}
                      </button>
                    )}

                    {r.staged_speed != null && r.staged_speed !== r.speed ? (
                      <span className="ml-auto flex-none font-mono text-[12px]">
                        <StagedValue
                          current={`${r.speed.toFixed(2)}×`}
                          staged={`${r.staged_speed.toFixed(2)}×`}
                          label={`Speed of the cue at ${fmtClock(r.start)}`}
                        />
                      </span>
                    ) : r.speed !== 1 ? (
                      <span
                        className="ml-auto flex-none font-mono text-[12px] text-ink3"
                        title="How fast this cue is spoken"
                      >
                        {r.speed.toFixed(2)}×
                      </span>
                    ) : null}

                    {was !== null ? (
                      <span className="flex-none text-[12px] text-blue-ink">
                        {drafts[r.id] !== undefined ? 'edited' : 'staged'}
                      </span>
                    ) : r.staged ? (
                      <span className="flex-none text-[12px] text-blue-ink">staged</span>
                    ) : null}

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        requestSeek(r.start);
                      }}
                      title={`Jump video to ${fmtClock(r.start)}`}
                      aria-label={`Seek to cue at ${fmtClock(r.start)}`}
                      className="flex-none rounded p-0.5 text-ink4 hover:text-brand
                                 transition-colors"
                    >
                      <span aria-hidden="true" className="text-[12px]">▶</span>
                    </button>

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onMute(r.id, !mutedIds.has(r.id));
                      }}
                      disabled={busy}
                      title={mutedIds.has(r.id) ? 'This cue is muted (staged) — click to unmute' : 'Mute this cue — it will be silent in the dub'}
                      aria-label={`${mutedIds.has(r.id) ? 'Unmute' : 'Mute'} cue at ${fmtClock(r.start)}`}
                      className={`flex-none rounded p-1 transition-colors disabled:opacity-30 ${
                        mutedIds.has(r.id) ? 'text-danger' : 'text-ink4 hover:text-danger'
                      }`}
                    >
                      {mutedIds.has(r.id) ? <IconMicOff /> : <IconMic />}
                    </button>
                  </div>

                  {was !== null && editing !== r.id ? (
                    <StagedText current={was} staged={shown} />
                  ) : null}

                  {showEnglish && r.src_text ? (
                    <p
                      className="mt-1 pl-[3.75rem] text-[12.5px] italic leading-snug text-ink4"
                      title={`Original ${sourceLangName} line (read-only)`}
                    >
                      {r.src_text}
                    </p>
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
