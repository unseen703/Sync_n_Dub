import * as Collapsible from '@radix-ui/react-collapsible';
import { useStore } from '../../store/useStore';
import { StageRibbon } from './StageRibbon';
import { Button } from '../frontend/Button';
import { Tooltip } from '../frontend/Tooltip';
import { IconCheck, IconAlert, IconSpinner, IconChevron, IconDownload } from '../frontend/Icons';
import { deriveOutcome, type OutcomeTone } from '../../lib/centre';
import { cn } from '../../lib/cn';
import { langClass } from '../../lib/lang';

const HIDDEN_SPEC_KEYS = new Set(['voices', 'tone', 'lipsync']);

export function ResultHeader() {
  const progress = useStore((s) => s.progress);
  const media = useStore((s) => s.media);
  const spec = useStore((s) => s.spec);
  const language = useStore((s) => s.speakers?.language ?? null);
  const outcome = deriveOutcome(progress, media);
  const showStatus = outcome.kind !== 'failed';

  const scrollToLipsync = () =>
    document
      .getElementById('lipsync-panel')
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const filteredLines = spec?.lines.filter((l) => !HIDDEN_SPEC_KEYS.has(l.key)) ?? [];

  return (
    <div className="space-y-3">
      {showStatus ? (
        <section className="card px-5 py-4">
          <div className="flex flex-wrap items-start gap-3">
            <StatusIcon tone={outcome.tone} />
            <div className="min-w-0 flex-1">
              <h2 className="text-[15px] font-semibold text-ink">{outcome.status}</h2>
              {outcome.sub ? (
                <p className="mt-0.5 text-[12.5px] text-ink3">{outcome.sub}</p>
              ) : null}
            </div>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              {outcome.action?.kind === 'sync-lips' ? (
                <Button variant="primary" onClick={scrollToLipsync}>
                  {outcome.action.label}
                </Button>
              ) : null}
              {outcome.action?.kind === 'download' && outcome.action.href ? (
                <DownloadLink href={outcome.action.href} primary>
                  {outcome.action.label}
                </DownloadLink>
              ) : null}
              {outcome.action?.kind === 'sync-lips' && outcome.dubDownloadUrl ? (
                <DownloadLink href={outcome.dubDownloadUrl}>Download dubbed video</DownloadLink>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}

      <StageRibbon />

      {spec ? (
        <Collapsible.Root className="card overflow-hidden">
          <Collapsible.Trigger className="group flex w-full items-center gap-1.5 px-5 py-3 text-[12px] text-ink3 transition-colors hover:text-ink">
            <IconChevron
              width={12}
              height={12}
              className="transition-transform group-data-[state=open]:rotate-180"
            />
            View job details
          </Collapsible.Trigger>
          <Collapsible.Content className="overflow-hidden">
            <div className="border-t border-line px-5 pb-4 pt-3">
              {spec.title ? (
                <h3 className="mb-3 text-[13.5px] font-semibold leading-snug text-ink">
                  {spec.title}
                </h3>
              ) : null}
              <dl className="space-y-2.5">
                {filteredLines.map((l) => (
                  <div key={l.key} className="grid grid-cols-[7rem_1fr] gap-2">
                    <dt className="pt-px text-[11.5px] text-ink4">{l.label}</dt>
                    <dd className="min-w-0">
                      <Tooltip label={l.hint ?? ''} side="right">
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
            </div>
          </Collapsible.Content>
        </Collapsible.Root>
      ) : null}
    </div>
  );
}

function DownloadLink({
  href,
  primary,
  children,
}: {
  href: string;
  primary?: boolean;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      download
      className={
        primary
          ? 'inline-flex h-9 items-center gap-2 rounded-xl bg-brand-fill px-3.5 text-[13px] font-medium text-white shadow-lg shadow-brand/25 transition-colors hover:bg-brand-soft'
          : 'inline-flex h-9 items-center gap-1.5 rounded-xl border border-edge/70 bg-surface/70 px-3 text-[12.5px] text-ink2 transition-colors hover:border-edge hover:text-ink'
      }
    >
      <IconDownload width={primary ? 14 : 13} height={primary ? 14 : 13} />
      {children}
    </a>
  );
}

function StatusIcon({ tone }: { tone: OutcomeTone }) {
  const map = {
    progress: { Icon: IconSpinner, cls: 'bg-brand/15 text-brand-ink' },
    bad: { Icon: IconAlert, cls: 'bg-danger/10 text-danger' },
    ok: { Icon: IconCheck, cls: 'bg-ok/10 text-ok' },
  }[tone];
  const { Icon, cls } = map;
  return (
    <span className={`flex h-9 w-9 flex-none items-center justify-center rounded-full ${cls}`}>
      <Icon width={18} height={18} />
    </span>
  );
}
