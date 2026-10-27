import * as Collapsible from '@radix-ui/react-collapsible';
import { useStore } from '../../store/useStore';
import { Eyebrow, Skeleton } from '../ui/Card';
import { IconChevron, IconClose, IconFilm } from '../ui/Icons';
import { Tooltip } from '../ui/Tooltip';
import { cn } from '../../lib/cn';
import { langClass } from '../../lib/lang';

/**
 * How this job was set up, in the user's words.
 *
 * Every line is rendered verbatim. `ui/spec.py` turns `mt_target_script=
 * codemix` into "Target language, English terms kept in English"
 * server-side on purpose: that mapping is domain knowledge that belongs
 * next to the pipeline, and duplicating it in TypeScript guarantees the two
 * drift. This component knows about `label`, `value`, `hint` and `tone` —
 * nothing else.
 */
export function SpecPanel({ onClose }: { onClose: () => void }) {
  const spec = useStore((s) => s.spec);
  const selected = useStore((s) => s.selected);
  const language = useStore((s) => s.speakers?.language ?? null);
  const admin = useStore((s) => s.mode === 'admin');

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-4 pb-3 pt-4">
        <Eyebrow>Job settings</Eyebrow>
        <button
          type="button"
          onClick={onClose}
          aria-label="Hide the job settings panel"
          className="rounded-lg p-1.5 text-ink3 transition-colors hover:bg-ink/[0.08] hover:text-ink"
        >
          <IconClose width={14} height={14} />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-6">
        {!selected ? (
          <p className="text-[12.5px] leading-relaxed text-ink3">
            Pick a dub from the list to see the language, the voices and the models it
            was run with.
          </p>
        ) : !spec ? (
          <div className="space-y-3">
            <Skeleton className="h-28" />
            <Skeleton className="h-3 w-3/4 rounded-full" />
            <Skeleton className="h-3 w-5/6 rounded-full" />
            <Skeleton className="h-3 w-2/3 rounded-full" />
          </div>
        ) : (
          <>
            {spec.thumbnail_url ? (
              <img
                src={spec.thumbnail_url}
                alt=""
                loading="lazy"
                referrerPolicy="no-referrer"
                className="mb-4 aspect-video w-full rounded-xl border border-line object-cover"
              />
            ) : (
              <div
                className="mb-4 grid aspect-video w-full place-items-center rounded-xl border
                           border-line bg-gradient-to-br from-surface to-canvas text-ink4"
              >
                <IconFilm width={22} height={22} />
              </div>
            )}

            {spec.title ? (
              <h2 className="mb-4 text-[13.5px] font-semibold leading-snug text-ink">
                {spec.title}
              </h2>
            ) : null}

            <dl className="space-y-3">
              {spec.lines.map((l) => (
                <div key={l.key} className="grid grid-cols-[6rem_1fr] gap-2">
                  <dt className="pt-px text-[11.5px] text-ink4">{l.label}</dt>
                  <dd className="min-w-0">
                    <Tooltip label={l.hint ?? ''} side="left">
                      <span
                        className={cn(
                          'break-words text-[12.5px] leading-snug',
                          l.tone === 'warn' ? 'text-warn' : 'text-ink2',
                          l.hint
                            && 'cursor-help decoration-edge decoration-dotted underline-offset-4 hover:underline',
                          (l.key === 'language' || l.key === 'script') && langClass(language),
                        )}
                      >
                        {l.value}
                      </span>
                    </Tooltip>
                  </dd>
                </div>
              ))}
            </dl>

            {/* The 2 a.m. view: kept, one click deep — and only in admin
                mode, where a wall of JSON is what was asked for rather than
                the first thing a new user trips over. */}
            {admin ? (
            <Collapsible.Root className="mt-5 border-t border-line pt-3">
              <Collapsible.Trigger
                className="group flex w-full items-center gap-1.5 text-[12px] text-ink3
                           transition-colors hover:text-ink"
              >
                <IconChevron
                  width={13}
                  height={13}
                  className="transition-transform group-data-[state=open]:rotate-180"
                />
                Advanced — the raw run config
              </Collapsible.Trigger>
              <Collapsible.Content className="overflow-hidden">
                <pre
                  className="mono mt-2 max-h-72 overflow-auto rounded-lg border border-line
                             bg-canvas/70 p-2.5 leading-relaxed text-ink3"
                >
                  {JSON.stringify(spec.raw_config, null, 2)}
                </pre>
              </Collapsible.Content>
            </Collapsible.Root>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
