import { useEffect, useRef, useState } from 'react';
import { useStore } from '../../store/useStore';
import { Button } from '../ui/Button';
import { Eyebrow } from '../ui/Card';
import { IconAlert, IconChevron, IconClose, IconSpinner } from '../ui/Icons';
import { fmtClock } from '../../lib/format';
import { LANGUAGES, type LangMeta } from '../../lib/lang';
import type {
  AgentAction,
  AgentScriptRow,
  AgentSpeaker,
  ToneOption,
  VoiceOption,
} from '../../api/types';

const GROUP_ORDER = [
  'Translation',
  'Voice',
  'Speakers',
  'Timing & audio',
  'Lip-sync & video',
  'Cleanup',
];

const PLACEHOLDER_RE = /<[^<>]+>/;

type PickOption = { value: string; label: string; group?: string };

/** Plain-language edit form. Inputs identify the job and available edit data; output is a staged edit request. */
export function ActionWorkspace({
  msg,
  setMsg,
  onSend,
  busy,
  error,
  warnings,
  actions,
  script,
  voiceOptions,
  toneOptions,
  speakers,
}: {
  msg: string;
  setMsg: (s: string) => void;
  onSend: () => void;
  busy: boolean;
  error: string | null;
  warnings: string[];
  actions: AgentAction[];
  script: AgentScriptRow[];
  voiceOptions: VoiceOption[];
  toneOptions: ToneOption[];
  speakers: AgentSpeaker[];
}) {
  const selectedCue = useStore((s) => s.selectedCue);
  const selectedSpeaker = useStore((s) => s.selectedSpeaker);
  const setSelectedCue = useStore((s) => s.setSelectedCue);
  const setSelectedSpeaker = useStore((s) => s.setSelectedSpeaker);
  const [showAll, setShowAll] = useState(false);

  const [pending, setPending] = useState<{ a: AgentAction; phrase: string } | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const cueRow = selectedCue ? script.find((r) => r.id === selectedCue) ?? null : null;

  const optionsFor = (a: AgentAction): PickOption[] | null => {
    const { needs } = a;

    if (a.options?.length) {
      return a.options.map((v) => ({ value: v, label: v }));
    }
    if (needs === 'voice') {
      return voiceOptions.map((v) => ({ value: v.voice, label: v.display }));
    }
    if (needs === 'tone') {
      return toneOptions.map((t) => ({ value: t.value, label: t.label, group: t.group }));
    }
    if (needs === 'speaker') {
      return speakers.map((sp) => ({ value: sp.id, label: sp.label || sp.id }));
    }
    if (needs === 'language') {
      return Object.entries(LANGUAGES)
        .filter(([, meta]) => (meta as LangMeta).script !== 'latn')
        .map(([, meta]) => ({
          value: (meta as LangMeta).english,
          label: `${(meta as LangMeta).english} — ${(meta as LangMeta).native}`,
        }))
        .sort((x, y) => x.value.localeCompare(y.value));
    }
    return null;
  };

  const selScope: 'segment' | 'speaker' | null = cueRow
    ? 'segment'
    : selectedSpeaker
      ? 'speaker'
      : null;
  const prefix = cueRow ? `cue ${cueRow.cue_no}` : selectedSpeaker ?? null;

  const fits = (a: AgentAction) =>
    !selScope || a.scopes.includes(selScope) || a.scopes.includes('all');

  const phraseFor = (a: AgentAction): string => {
    if (prefix && selScope && a.scopes.includes(selScope) && a.fill) {
      return `${prefix}: ${a.fill}`;
    }
    return a.examples[0] ?? a.fill ?? a.title;
  };

  const pick = (a: AgentAction) => {
    const phrase = phraseFor(a);
    const m = PLACEHOLDER_RE.exec(phrase);
    if (!m) {
      setMsg(phrase);
      setPending(null);
      return;
    }
    const opts = optionsFor(a);
    if (opts && opts.length) {

      setMsg(phrase);
      setPending({ a, phrase });
      return;
    }

    setMsg(phrase);
    setPending(null);
    requestAnimationFrame(() => {
      const el = inputRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(m.index, m.index + m[0].length);
    });
  };

  const fillPending = (value: string) => {
    if (!pending) return;
    setMsg(pending.phrase.replace(PLACEHOLDER_RE, value));
    setPending(null);
  };

  const unfilled = PLACEHOLDER_RE.exec(msg);

  const quick = actions.filter((a) => a.quick);

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
          Tip: click a cue or a speaker to aim these at it.
        </p>
      )}

      <div className="flex gap-2">
        <input
          ref={inputRef}
          value={msg}
          onChange={(e) => {
            setMsg(e.target.value);
            setPending(null);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !unfilled) onSend();
          }}
          placeholder='e.g. "make SPEAKER_00 female and speed up 0:30 to 0:45"'
          aria-label="Describe the change"
          className="min-w-0 flex-1 rounded-lg border border-edge/70 bg-canvas/60 px-3 py-2
                     text-[13px] text-ink placeholder:text-ink4"
        />
        <Button
          onClick={onSend}
          disabled={busy || !msg.trim() || Boolean(unfilled)}
          title={unfilled ? `Replace ${unfilled[0]} first` : undefined}
        >
          {busy ? <IconSpinner /> : null} Send
        </Button>
      </div>

      {pending ? (
        <div className="mt-2">
          <ValuePicker
            label={`Pick a ${pending.a.needs}`}
            options={optionsFor(pending.a) ?? []}
            onPick={fillPending}
            onDismiss={() => setPending(null)}
            disabled={busy}
          />
        </div>
      ) : null}

      {unfilled && !pending ? (
        <p className="mt-2 text-[11.5px] text-ink4">
          Replace <span className="font-mono text-ink3">{unfilled[0]}</span> before sending —
          the agent will not guess a value you did not give it.
        </p>
      ) : null}

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

function ValuePicker({
  label,
  options,
  onPick,
  onDismiss,
  disabled,
}: {
  label: string;
  options: PickOption[];
  onPick: (value: string) => void;
  onDismiss: () => void;
  disabled: boolean;
}) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onDoc = (ev: MouseEvent) => {
      if (ref.current && !ref.current.contains(ev.target as Node)) onDismiss();
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [onDismiss]);

  const groups = [...new Set(options.map((o) => o.group ?? ''))];

  return (
    <div
      ref={ref}
      className="rounded-lg border border-brand/30 bg-brand/5 px-3 py-2"
    >
      <div className="mb-1.5 flex items-center gap-2">
        <span className="text-[11.5px] text-ink3">{label}</span>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss the picker"
          className="ml-auto flex items-center gap-1 text-[11px] text-ink4
                     transition-colors hover:text-ink"
        >
          <IconClose width={11} height={11} /> cancel
        </button>
      </div>
      <div role="listbox" className="max-h-56 overflow-auto">
        {groups.map((g) => (
          <div key={g}>
            {g ? (
              <div className="mb-1 mt-1 text-[10.5px] font-semibold uppercase
                              tracking-wide text-ink4">
                {g}
              </div>
            ) : null}
            <div className="flex flex-wrap gap-1.5">
              {options
                .filter((o) => (o.group ?? '') === g)
                .map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    role="option"
                    disabled={disabled}
                    onClick={() => onPick(o.value)}
                    className="rounded-lg border border-line bg-surface/60 px-2.5
                               py-1 text-[12px] text-ink2 transition-colors
                               hover:border-edge hover:text-ink disabled:opacity-50"
                  >
                    {o.label}
                  </button>
                ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
