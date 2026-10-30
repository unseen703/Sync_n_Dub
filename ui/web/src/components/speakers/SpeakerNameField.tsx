import { useEffect, useRef, useState } from 'react';
import { cn } from '../../lib/cn';
import { IconPencil } from '../ui/Icons';
import { Tooltip } from '../ui/Tooltip';
import { useStore } from '../../store/useStore';
import type { Speaker } from '../../api/types';

export function SpeakerNameField({ speaker }: { speaker: Speaker }) {
  const rename = useStore((s) => s.renameSpeaker);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(speaker.label ?? '');
  const [saving, setSaving] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editing) setDraft(speaker.label ?? '');
  }, [speaker.label, editing]);

  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  const commit = async () => {
    const next = draft.trim();
    setEditing(false);
    if (next === (speaker.label ?? '')) return;
    setSaving(true);
    try {
      await rename(speaker.id, next);
    } finally {
      setSaving(false);
    }
  };

  if (editing) {
    return (
      <input
        ref={input}
        value={draft}
        maxLength={80}
        aria-label={`Name for ${speaker.id}`}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => void commit()}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            void commit();
          } else if (e.key === 'Escape') {
            e.preventDefault();
            setDraft(speaker.label ?? '');
            setEditing(false);
          }
          e.stopPropagation();
        }}
        placeholder={speaker.id}
        className="field h-8 w-44 rounded-lg px-2 text-[15px]"
      />
    );
  }

  return (
    <Tooltip label={speaker.label ? `Also known as ${speaker.id}` : 'Give this speaker a name'}>
      <button
        type="button"
        onClick={() => setEditing(true)}
        aria-label={
          speaker.label
            ? `Rename ${speaker.label} (currently ${speaker.id})`
            : `Name ${speaker.id}`
        }
        className="group -ml-1 flex min-w-0 items-center gap-1.5 rounded-lg px-1 py-0.5
                   transition-colors hover:bg-ink/[0.07]"
      >
        <span
          className={cn(
            'truncate text-[15px]',
            speaker.label ? 'font-semibold text-ink' : 'font-mono text-[13px] text-ink3',
            saving && 'opacity-60',
          )}
        >
          {speaker.label ?? speaker.id}
        </span>
        <IconPencil
          width={12}
          height={12}
          className="shrink-0 text-ink4 opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100"
        />
      </button>
    </Tooltip>
  );
}
