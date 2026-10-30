import { m } from 'framer-motion';
import { Button } from '../ui/Button';
import { IconAlert } from '../ui/Icons';
import { useStore } from '../../store/useStore';
import type { Failure } from '../../api/types';

/** Failure message. Input: normalized failure data. Output: an actionable error card or nothing. */
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
