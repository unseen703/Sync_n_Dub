import { useEffect, useRef, useState } from 'react';
import { useStore } from '../../store/useStore';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconChevron, IconClose, IconSpinner } from '../ui/Icons';
import { fmtClock } from '../../lib/format';
import { LANGUAGES, type LangMeta } from '../../lib/lang';
import type { AgentAction, AgentScriptRow } from '../../api/types';

/**
 * The one place you ask for a change — the merger of the old "Ask for a
 * change" box, "Things you can say" and "What it can do".
 *
 * Those three overlapped: "Things you can say" was just each op's example
 * phrasing, and "What it can do" was the same ops described. Here each action
 * carries both halves on one card, and clicking it drops a ready phrasing into
 * the request box.
 *
 * Context-awareness without syntax: a clicked cue or speaker (held in the
 * store) sets the SELECTION, and an action that fits it is phrased against it
 * automatically — "cue 5: speed this up by 1.2x" — so the user never types a
 * timestamp or a speaker id. Everything that decides grouping, which actions
 * are "quick", and how each is phrased is served by the backend
 * (`backend/edits.py`), so this panel cannot advertise an op the parser does
 * not have.
 */

// The display order of the intent groups. The backend sends the SAME strings
// (`USER_ACTIONS[...]['category']`); anything not listed here is appended
// rather than dropped, so a new category never silently disappears.
const GROUP_ORDER = [
  'Translation',
  'Voice',
  'Speakers',
  'Timing & audio',
  'Lip-sync & video',
  'Cleanup',
];

export function ActionWorkspace({
  msg,
  setMsg,
  onSend,
  busy,
  error,
  warnings,
  actions,
  script,
}: {
  msg: string;
  setMsg: (s: string) => void;
  onSend: () => void;
  busy: boolean;
  error: string | null;
  warnings: string[];
  actions: AgentAction[];
  script: AgentScriptRow[];
}) {
  const selectedCue = useStore((s) => s.selectedCue);
  const selectedSpeaker = useStore((s) => s.selectedSpeaker);
  const setSelectedCue = useStore((s) => s.setSelectedCue);
  const setSelectedSpeaker = useStore((s) => s.setSelectedSpeaker);
  const [showAll, setShowAll] = useState(false);

  const cueRow = selectedCue ? script.find((r) => r.id === selectedCue) ?? null : null;

  // The scope the current selection provides, and the prefix the agent accepts
  // for it. `cue N` and a bare speaker id are both documented vocabulary.
  const selScope: 'segment' | 'speaker' | null = cueRow
    ? 'segment'
    : selectedSpeaker
      ? 'speaker'
      : null;
  const prefix = cueRow ? `cue ${cueRow.cue_no}` : selectedSpeaker ?? null;

  const fits = (a: AgentAction) =>
    !selScope || a.scopes.includes(selScope) || a.scopes.includes('all');

  // A concrete, scope-pinned phrase when the selection fits the op; the op's
  // own worked example otherwise.
  const phraseFor = (a: AgentAction): string => {
    if (prefix && selScope && a.scopes.includes(selScope) && a.fill) {
      return `${prefix}: ${a.fill}`;
    }
    return a.examples[0] ?? a.fill ?? a.title;
  };

  const pick = (a: AgentAction) => setMsg(phraseFor(a));

  const quick = actions.filter((a) => a.quick);
  // Fold the selection-relevant ones to the front when something is selected.
  const quickSorted = selScope
    ? [...quick].sort((x, y) => Number(fits(y)) - Number(fits(x)))
    : quick;

  const known = GROUP_ORDER.map((g) => ({
    g,
    items: actions.filter((a) => a.category === g),
  })).filter((x) => x.items.length);
  const extraNames = [...new Set(actions.map((a) => a.category))].filter(
    (c) => !GROUP_ORDER.includes(c),
  );
  const groups = [
    ...known,
    ...extraNames.map((g) => ({ g, items: actions.filter((a) => a.category === g) })),
  ];

  return (
    <section className="card p-4">
      <Eyebrow className="mb-2">Ask for a change</Eyebrow>

      {/* What the actions below will aim at. */}
      {selScope ? (
        <div className="mb-2.5 flex items-center gap-2 rounded-lg border border-brand/30 bg-brand/5 px-3 py-1.5 text-[12px]">
          <span className="text-ink3">Editing</span>
          <span className="min-w-0 truncate font-medium text-ink">
            {cueRow ? `cue #${cueRow.cue_no} · ${fmtClock(cueRow.start)}` : selectedSpeaker}
          </span>
          <button
            type="button"
            onClick={() => {
              setSelectedCue(null);
              setSelectedSpeaker(null);
            }}
            aria-label="Clear the selection"
            className="ml-auto flex flex-none items-center gap-1 text-ink4 transition-colors hover:text-ink"
          >
            <IconClose width={12} height={12} /> clear
          </button>
        </div>
      ) : (
        <p className="mb-2.5 text-[11.5px] text-ink4">
          Tip: click a cue or a speaker to aim these at it — no timestamps or ids to type.
        </p>
      )}

      {/* The free-text request. */}
      <div className="flex gap-2">
        <input
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) onSend();
          }}
          placeholder='e.g. "make SPEAKER_00 female and speed up 0:30 to 0:45"'
          aria-label="Describe the change"
          className="min-w-0 flex-1 rounded-lg border border-edge/70 bg-canvas/60 px-3 py-2
                     text-[13px] text-ink placeholder:text-ink4"
        />
        <Button onClick={onSend} disabled={busy || !msg.trim()}>
          {busy ? <IconSpinner /> : null} Send
        </Button>
      </div>

      {/* Target-language picker — one of the 22 Indic languages the home
          page already advertises. Clicking a language fills the request box
          with "change the target language to <English>"; the user hits Send
          (so a misclick stays reviewable). The source table is
          `ui/web2/src/lib/lang.ts::LANGUAGES`, which is the TS mirror of
          `ui/langs.py::LANGUAGES` — one list for home-page, chat composer
          and backend handler. */}
      <div className="mt-2">
        <LanguagePicker
          disabled={busy}
          onPick={(_iso, english) =>
            setMsg(`change the target language to ${english}`)
          }
        />
      </div>

      {error ? (
        <p className="mt-2 flex items-start gap-1.5 text-[12.5px] text-warn">
          <IconAlert width={13} height={13} className="mt-0.5 flex-none" />
          <span>{error}</span>
        </p>
      ) : null}

      {warnings.map((w, i) => (
        <p key={i} className="mt-2 text-[12px] text-ink3">
          {w}
        </p>
      ))}

      {/* The curated common actions. */}
      {quickSorted.length ? (
        <div className="mt-3">
          <div className="mb-1.5 text-[11px] text-ink4">or pick a common action</div>
          <div className="flex flex-wrap gap-1.5">
            {quickSorted.map((a) => (
              <button
                key={a.op}
                type="button"
                onClick={() => pick(a)}
                title={a.changes}
                className={`rounded-lg border px-2.5 py-1.5 text-[12px] transition-colors ${
                  selScope && fits(a)
                    ? 'border-brand/40 bg-brand/10 text-ink hover:bg-brand/15'
                    : 'border-line bg-surface/50 text-ink2 hover:border-edge hover:text-ink'
                }`}
              >
                {a.title}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {/* The long tail, grouped by intent. */}
      {actions.length ? (
        <>
          <button
            type="button"
            onClick={() => setShowAll((v) => !v)}
            className="group mt-3 flex items-center gap-1.5 text-[12px] text-ink3 transition-colors hover:text-ink"
          >
            <IconChevron
              width={12}
              height={12}
              className={`transition-transform ${showAll ? 'rotate-180' : ''}`}
            />
            {showAll ? 'Hide full list' : 'See all actions'}
          </button>

          {showAll ? (
            <div className="mt-3 space-y-4">
              {groups.map(({ g, items }) => (
                <div key={g}>
                  <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-ink4">
                    {g}
                  </div>
                  <ul className="space-y-1.5">
                    {items.map((a) => {
                      const dim = Boolean(selScope) && !fits(a);
                      return (
                        <li key={a.op}>
                          <button
                            type="button"
                            onClick={() => pick(a)}
                            className={`w-full rounded-lg border border-line bg-surface/40 px-3 py-2
                                        text-left transition-colors hover:border-edge hover:bg-surface ${
                                          dim ? 'opacity-55' : ''
                                        }`}
                          >
                            <div className="text-[12.5px] font-medium text-ink">{a.title}</div>
                            <div className="mt-0.5 text-[11.5px] text-ink3">
                              <span className="text-ink4">You give it </span>
                              {a.takes}
                            </div>
                            <div className="mt-0.5 text-[11.5px] text-ink3">
                              <span className="text-ink4">Then </span>
                              {a.changes}
                            </div>
                            <div className="mt-1 text-[11px] text-ink4">Re-runs: {a.replays}</div>
                            {a.examples[0] ? (
                              <div className="mt-1 text-[11px] italic text-ink4">
                                e.g. &ldquo;{a.examples[0]}&rdquo;
                              </div>
                            ) : null}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

/**
 * One of the 22 Indic languages, listed as `English — Native`.
 *
 * The list is the home-page dropdown's: every entry of `LANGUAGES` whose
 * script is not `latn`, sorted by English name. Clicking a row drops a
 * fully-formed request into the chat input (`change the target language to
 * <English>`) rather than firing the request itself — a misclick stays
 * reviewable before Send. The regex rule parser recognises that phrasing
 * and emits a `set_target_language` op; the system prompt lists the op too,
 * so the LLM path handles the same message when a key is configured.
 */
function LanguagePicker({
  onPick,
  disabled,
}: {
  onPick: (iso: string, english: string, native: string) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  // Close on outside click. Keeping it inline rather than reaching for a
  // portal — this picker lives in a card the composer already owns.
  useEffect(() => {
    if (!open) return;
    const onDoc = (ev: MouseEvent) => {
      if (ref.current && !ref.current.contains(ev.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const indic = Object.entries(LANGUAGES)
    .filter(([, meta]) => (meta as LangMeta).script !== 'latn')
    .map(([iso, meta]) => ({ iso, ...(meta as LangMeta) }))
    .sort((a, b) => a.english.localeCompare(b.english));

  return (
    <div ref={ref} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        className="inline-flex items-center gap-1.5 rounded-lg border border-line
                   bg-surface/60 px-2.5 py-1.5 text-[12px] text-ink2
                   transition-colors hover:border-edge hover:text-ink
                   disabled:opacity-50"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        Change language
        <IconChevron width={11} height={11} />
      </button>

      {open ? (
        <ul
          role="listbox"
          className="absolute z-20 mt-1 max-h-72 w-64 overflow-auto rounded-lg
                     border border-edge/70 bg-surface py-1 shadow-lg"
        >
          {indic.map((row) => (
            <li key={row.iso}>
              <button
                type="button"
                role="option"
                onClick={() => {
                  onPick(row.iso, row.english, row.native);
                  setOpen(false);
                }}
                className="flex w-full items-baseline justify-between gap-3
                           px-3 py-1.5 text-left text-[12.5px] text-ink2
                           hover:bg-canvas/60 hover:text-ink"
              >
                <span className="font-medium">{row.english}</span>
                <span className={`text-[12px] text-ink3 lang-${row.script}`}>
                  {row.native}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
