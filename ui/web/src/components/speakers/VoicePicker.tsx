import { useEffect, useMemo, useState } from 'react';
import * as Popover from '@radix-ui/react-popover';
import { cn } from '../../lib/cn';
import { IconChevron, IconCheck } from '../ui/Icons';
import { Eyebrow } from '../ui/Card';
import { genderWord } from '../../lib/format';
import { langPriority } from '../../lib/langGroups';
import type { Speaker, VoiceOption } from '../../api/types';

export function VoicePicker({
  speaker,
  options,
  value,
  onChange,
  disabled,
  target,
}: {
  speaker: Speaker;
  options: VoiceOption[];
  value: string | null;
  onChange: (voice: string) => void;
  disabled?: boolean;
  target: string | null;
}) {
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const recommended = useMemo(
    () => options.filter((o) => o.recommended_for?.includes(speaker.id)),
    [options, speaker.id],
  );

  const hasRecommendation = recommended.length > 0;

  useEffect(() => {
    if (open) setShowAll(!hasRecommendation);
  }, [open, hasRecommendation]);

  const shown = showAll || !hasRecommendation ? options : recommended;
  const ordered = useMemo(
    () =>
      [...shown].sort(
        (a, b) => langPriority(a.language, target) - langPriority(b.language, target),
      ),
    [shown, target],
  );
  const grouped = useMemo(() => groupByGender(ordered), [ordered]);

  const selected = options.find((o) => o.voice === value);
  const label = selected?.display ?? (value ? value.split('-').pop() : 'Not assigned');

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger
        disabled={disabled || options.length === 0}
        aria-label={`Voice for ${speaker.label ?? speaker.id}`}
        className={cn(
          'inline-flex h-8 min-w-[9.5rem] items-center justify-between gap-2 rounded-lg border px-2.5',
          'text-[13px] transition-colors disabled:cursor-not-allowed disabled:opacity-50',
          'border-edge/70 bg-surface/70 text-ink hover:border-edge',
        )}
      >
        <span className={cn('truncate', !value && 'text-ink4')}>{label}</span>
        <IconChevron width={13} height={13} className="shrink-0 text-ink3" />
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 flex max-h-[22rem] w-72 animate-slide-up flex-col overflow-hidden
                     rounded-xl border border-line bg-raised shadow-2xl shadow-black/70"
        >
          {options.length === 0 ? (
            <p className="px-3 py-4 text-[12.5px] leading-relaxed text-ink3">
              This engine does not publish a voice list, so the adapter&rsquo;s own default is
              used for every speaker.
            </p>
          ) : (
            <>
              <div className="border-b border-line px-3 py-2.5">
                <div className="text-[12.5px] font-medium text-ink">
                  {showAll || !hasRecommendation
                    ? `All ${options.length} voices`
                    : `Suggested for ${speaker.label ?? speaker.id}`}
                </div>
                <p className="mt-0.5 text-[11.5px] leading-snug text-ink3">
                  {hasRecommendation
                    ? showAll
                      ? 'Every voice this engine offers in the target language.'
                      : `Matched to the ${genderWord(speaker.gender)?.toLowerCase()} voice detected for this speaker — a guess, not a rule.`
                    : 'No gender was detected for this speaker, so nothing is singled out.'}
                </p>
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
                {grouped.map(([title, rows]) =>
                  rows.length === 0 ? null : (
                    <div key={title} className="mb-1 last:mb-0">
                      {grouped.filter(([, r]) => r.length).length > 1 ? (
                        <Eyebrow className="px-2 py-1.5">{title}</Eyebrow>
                      ) : null}
                      {rows.map((o) => {
                        const isSel = o.voice === value;
                        return (
                          <button
                            key={o.voice}
                            type="button"
                            onClick={() => {
                              onChange(o.voice);
                              setOpen(false);
                            }}
                            aria-pressed={isSel}
                            className={cn(
                              'flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left',
                              'transition-colors',
                              isSel ? 'bg-brand/25' : 'hover:bg-ink/[0.07]',
                            )}
                          >
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-1.5">
                                <span
                                  className={cn(
                                    'truncate text-[13px]',
                                    isSel ? 'font-semibold text-ink' : 'text-ink2',
                                  )}
                                >
                                  {o.display}
                                </span>
                                {o.tier === 'lead' ? (
                                  <span
                                    className="rounded border border-aqua/35 bg-aqua/10 px-1
                                               text-[10px] font-semibold uppercase tracking-wider text-aqua-ink"
                                  >
                                    lead
                                  </span>
                                ) : null}
                              </div>
                              <div className="truncate text-[11.5px] text-ink4">
                                {[genderWord(o.gender), o.age, o.language]
                                  .filter(Boolean)
                                  .join(' · ') || o.voice}
                              </div>
                            </div>
                            {isSel ? (
                              <IconCheck width={14} height={14} className="shrink-0 text-aqua-ink" />
                            ) : null}
                          </button>
                        );
                      })}
                    </div>
                  ),
                )}
              </div>

              {hasRecommendation ? (
                <button
                  type="button"
                  onClick={() => setShowAll((v) => !v)}
                  className="border-t border-line px-3 py-2.5 text-left text-[12.5px]
                             font-medium text-brand-ink transition-colors hover:bg-ink/[0.05]"
                >
                  {showAll
                    ? `Back to the ${recommended.length} suggested`
                    : `Show all ${options.length} voices`}
                </button>
              ) : null}
            </>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function groupByGender(rows: VoiceOption[]): Array<[string, VoiceOption[]]> {
  return [
    ['Female', rows.filter((o) => o.gender === 'female')],
    ['Male', rows.filter((o) => o.gender === 'male')],
    ['Unlabelled', rows.filter((o) => o.gender !== 'female' && o.gender !== 'male')],
  ];
}
