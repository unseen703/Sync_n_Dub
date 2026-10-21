import * as Collapsible from '@radix-ui/react-collapsible';
import { m } from 'framer-motion';
import { Button } from '../ui/Button';
import { IconAlert, IconChevron } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import type { Failure } from '../../api/types';

/**
 * What went wrong, said in a sentence the person who pasted the link can
 * act on.
 *
 * This renders `progress.failure`, never `progress.error`. `ui/errors.py`
 * matches the traceback against the failures this pipeline actually
 * produces and answers with what happened, whose problem it is and what to
 * do next; `error` is the raw last line of a traceback and belongs nowhere
 * near a user-facing screen.
 *
 * `raw` is the one concession, and it is deliberately bounded: a single
 * line, behind a closed disclosure, never a log stream. The runner log view
 * this app used to carry was deleted outright — operators read the file on
 * disk, and everyone else gets a sentence.
 *
 * `recognised: false` is stated rather than hidden. "We do not recognise
 * this one" is more useful than a confident-sounding paraphrase of a line
 * we did not understand.
 */
export function FailureCard({
  failure,
  cancelled,
}: {
  failure: Failure | null;
  cancelled: boolean;
}) {
  const setNewJobOpen = useStore((s) => s.setNewJobOpen);

  if (cancelled) {
    return (
      <Banner tone="warn" title="This dub was stopped">
        <p className="t-body">
          Someone stopped this run before it finished. Everything it had already produced is
          still on disk and still playable below.
        </p>
      </Banner>
    );
  }

  if (!failure) return null;

  return (
    <Banner tone="danger" title={failure.title}>
      <p className="t-body">{failure.detail}</p>

      {failure.action ? (
        <p className="mt-2 text-[13px] leading-relaxed text-ink">
          <span className="font-semibold">What to do: </span>
          {failure.action}
        </p>
      ) : null}

      {!failure.recognised ? (
        <p className="mt-2 text-[12.5px] leading-relaxed text-ink3">
          This is not a failure we recognise, so the line above is the error itself rather than
          an explanation of it.
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={() => setNewJobOpen(true)}>
          Start it again
        </Button>
      </div>

      {failure.raw ? (
        <Collapsible.Root className="mt-3 border-t border-danger/25 pt-2.5">
          <Collapsible.Trigger
            className="group flex items-center gap-1.5 text-[12px] text-ink3
                       transition-colors hover:text-ink"
          >
            <IconChevron
              width={12}
              height={12}
              className="transition-transform group-data-[state=open]:rotate-180"
            />
            Technical details
          </Collapsible.Trigger>
          <Collapsible.Content className="overflow-hidden">
            <p className="mono mt-2 break-words rounded-lg border border-line bg-canvas/70 px-2.5 py-2 text-ink3">
              {failure.raw}
            </p>
          </Collapsible.Content>
        </Collapsible.Root>
      ) : null}
    </Banner>
  );
}

function Banner({
  tone,
  title,
  children,
}: {
  tone: 'danger' | 'warn';
  title: string;
  children: React.ReactNode;
}) {
  const box =
    tone === 'danger'
      ? 'border-danger/45 bg-danger/[0.07]'
      : 'border-warn/40 bg-warn/[0.07]';
  const icon = tone === 'danger' ? 'text-danger' : 'text-warn';
  return (
    <m.section
      role="alert"
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 rounded-2xl border px-4 py-3.5 ${box}`}
    >
      <IconAlert className={`mt-0.5 shrink-0 ${icon}`} width={18} height={18} />
      <div className="min-w-0 flex-1">
        <h2 className="text-[14px] font-semibold text-ink">{title}</h2>
        <div className="mt-1">{children}</div>
      </div>
    </m.section>
  );
}
